# SPDX-License-Identifier: GPL-2.0-or-later
"""Version 1 little-endian surface cache. No pickle or executable payloads."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

MAGIC = b'K3DREND\0'
VERSION = 1
HEADER = struct.Struct('<8sIIQ32s')
RECORD = struct.Struct('<18dIII')
MAX_BYTES = 256 * 1024 * 1024
MAX_METADATA = 8 * 1024 * 1024
ALGORITHM = 'centroid-lambert-point-one-bounce-v1'


def _path(path):
    path = Path(path)
    if path.suffix.lower() != '.rend':
        raise ValueError('Lighting cache path must end in .REND')
    return path


def save(path, cache):
    from . import EPS, FAR
    path = _path(path)
    if cache.dirty or cache.transport_dirty or cache.entries.keys() != cache.patches.keys():
        raise ValueError('Finish lighting updates before saving a .REND cache')
    keys = list(cache.patches)
    indices = {k: i for i, k in enumerate(keys)}
    metadata = json.dumps({'algorithm': ALGORITHM, 'samples': cache.samples,
                           'epsilon': EPS, 'far': FAR, 'lights': cache.lights,
                           'keys': [[k[0], k[1].hex(), k[2]] for k in keys]},
                          separators=(',', ':'), allow_nan=False).encode('utf8')
    payload = bytearray()
    for key in keys:
        patch, entry = cache.patches[key], cache.entries[key]
        values = [v for vertex in patch.vertices for v in vertex]
        values += list(patch.color) + list(entry['direct']) + list(entry['radiance'])
        links, segments = entry['links'], entry['segments']
        payload.extend(RECORD.pack(*values, len(links), len(segments), len(entry['transport_segments'])))
        payload.extend(np.asarray([indices[k] for k in links], dtype='<u4').tobytes())
        payload.extend(np.asarray(segments, dtype='<f8').tobytes())
        if len(payload) > MAX_BYTES:
            raise ValueError('.REND cache exceeds the 256 MiB prototype limit')
    if len(metadata) > MAX_METADATA or len(metadata) + len(payload) > MAX_BYTES:
        raise ValueError('.REND cache exceeds prototype size limits')
    digest = hashlib.sha256(metadata + payload).digest()
    # Atomic replacement preserves the last complete cache if a write fails.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as out:
            temp = out.name
            out.write(HEADER.pack(MAGIC, VERSION, len(metadata), len(payload), digest))
            out.write(metadata)
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)


def load(path):
    from . import Cache, Patch, EPS, FAR
    path = _path(path)
    with path.open('rb') as source:
        raw_header = source.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError('Truncated .REND header')
        magic, version, meta_size, data_size, digest = HEADER.unpack(raw_header)
        if magic != MAGIC or version != VERSION:
            raise ValueError('Unsupported .REND format or version')
        if meta_size > MAX_METADATA or meta_size + data_size > MAX_BYTES:
            raise ValueError('.REND exceeds prototype size limits')
        body = source.read(meta_size + data_size + 1)
        if len(body) != meta_size + data_size:
            raise ValueError('Truncated .REND data or unexpected trailing bytes')
        if hashlib.sha256(body).digest() != digest:
            raise ValueError('.REND checksum mismatch')
    try:
        meta = json.loads(body[:meta_size])
        if (meta['algorithm'], meta['epsilon'], meta['far']) != (ALGORITHM, EPS, FAR):
            raise ValueError('Incompatible lighting estimator')
        samples = meta['samples']
        if type(samples) is not int or not 1 <= samples <= 4096 or len(meta['keys']) > 100000:
            raise ValueError('Invalid sample or patch count')
        keys = []
        for name, topology, index in meta['keys']:
            topology = bytes.fromhex(topology)
            if not isinstance(name, str) or len(topology) != 32 or type(index) is not int or index < 0:
                raise ValueError('Invalid surface identity')
            keys.append((name, topology, index))
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate surface identity')
        lights = tuple((tuple(p), tuple(c)) for p, c in meta['lights'])
        if lights and (np.asarray(lights).shape != (len(lights), 2, 3) or
                       not np.isfinite(lights).all() or np.any(np.asarray(lights)[:, 1] < 0)):
            raise ValueError('Invalid light data')
        offset, patches, entries = meta_size, {}, {}
        for key in keys:
            fields = RECORD.unpack_from(body, offset)
            offset += RECORD.size
            values = np.asarray(fields[:18])
            link_count, segment_count, transport_count = fields[18:]
            if (not np.isfinite(values).all() or np.any(values[9:] < 0) or
                    link_count > samples or transport_count != samples or
                    not samples <= segment_count <= samples + len(lights)):
                raise ValueError('Invalid patch or dependency data')
            size = link_count * 4 + segment_count * 48
            if offset + size > len(body):
                raise ValueError('Truncated dependency data')
            links = np.frombuffer(body, dtype='<u4', count=link_count, offset=offset)
            if np.any(links >= len(keys)):
                raise ValueError('Invalid connection index')
            offset += link_count * 4
            segments = np.frombuffer(body, dtype='<f8', count=segment_count * 6, offset=offset).reshape(-1, 2, 3)
            if not np.isfinite(segments).all():
                raise ValueError('Invalid ray segment')
            offset += segment_count * 48
            segments = [(tuple(a), tuple(b)) for a, b in segments]
            patches[key] = Patch(key, tuple(tuple(v) for v in values[:9].reshape(3, 3)), tuple(values[9:12]))
            entries[key] = {'direct': values[12:15].copy(), 'radiance': values[15:18].copy(),
                            'links': [keys[i] for i in links], 'segments': segments,
                            'transport_segments': segments[-transport_count:]}
        if offset != len(body):
            raise ValueError('Unexpected .REND record data')
    except (KeyError, TypeError, IndexError, struct.error, UnicodeError) as exc:
        raise ValueError('Malformed .REND cache') from exc
    cache = Cache(samples)
    cache.sync(patches, lights)  # Rebuild only the acceleration structure; zero lighting rays.
    cache.entries = entries
    cache.dirty.clear()
    cache.transport_dirty.clear()
    cache.last_invalidated = 0
    return cache
