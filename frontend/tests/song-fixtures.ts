import { zipSync, strToU8 } from 'fflate'

// An original, silent map fixture. It is not a bundled song or a policy performance.
export function wav(seconds = 20) {
  const rate = 8000
  const bytes = new Uint8Array(44 + rate * seconds * 2)
  const view = new DataView(bytes.buffer)
  const write = (offset: number, text: string) =>
    [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)))
  write(0, 'RIFF')
  view.setUint32(4, bytes.length - 8, true)
  write(8, 'WAVEfmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(36, 'data')
  view.setUint32(40, bytes.length - 44, true)
  return bytes
}

export const info = {
  _version: '2.0.0',
  _songName: 'Orbit Test',
  _songSubName: 'Test chart',
  _songAuthorName: 'Test Artist',
  _levelAuthorName: 'Test Mapper',
  _beatsPerMinute: 120,
  _songTimeOffset: 0,
  _songFilename: 'song.wav',
  _coverImageFilename: '',
  _difficultyBeatmapSets: [
    {
      _beatmapCharacteristicName: 'Standard',
      _difficultyBeatmaps: [
        {
          _difficulty: 'Normal',
          _beatmapFilename: 'Normal.dat',
          _noteJumpMovementSpeed: 10,
          _noteJumpStartBeatOffset: 0,
        },
        {
          _difficulty: 'Hard',
          _beatmapFilename: 'Hard.dat',
          _noteJumpMovementSpeed: 14,
          _noteJumpStartBeatOffset: 0,
        },
      ],
    },
  ],
}

export const v2 = {
  _version: '2.0.0',
  _notes: [
    { _time: 4, _lineIndex: 3, _lineLayer: 1, _type: 1, _cutDirection: 8 },
    { _time: 5, _lineIndex: 0, _lineLayer: 1, _type: 0, _cutDirection: 8 },
    { _time: 8, _lineIndex: 2, _lineLayer: 2, _type: 3 },
    { _time: 16, _lineIndex: 1, _lineLayer: 0, _type: 0, _cutDirection: 5 },
  ],
  _obstacles: [{ _time: 10, _duration: 2, _lineIndex: 0, _width: 1, _type: 1 }],
}

export const v3 = {
  version: '3.3.0',
  colorNotes: [{ b: 4, x: 3, y: 1, c: 1, d: 8 }, { b: 5 }, { b: 16, x: 1, d: 5 }],
  bombNotes: [{ b: 8, x: 2, y: 2 }],
  obstacles: [{ b: 10, d: 2, x: 0, y: 2, w: 1, h: 3 }],
  bpmEvents: [{ b: 4, m: 60 }],
}

export function mapZip(overrides: Record<string, Uint8Array> = {}) {
  const files = {
    'Info.dat': strToU8(JSON.stringify(info)),
    'Normal.dat': strToU8(JSON.stringify(v2)),
    'Hard.dat': strToU8(JSON.stringify(v3)),
    'song.wav': wav(),
    ...overrides,
  }
  return zipSync(files)
}

export const result = {
  id: 'abcd',
  metadata: {
    songName: 'Orbit Test',
    songAuthorName: 'Test Artist',
    levelAuthorName: 'Test Mapper',
    bpm: 120,
    duration: 20,
  },
  versions: [
    {
      state: 'Published',
      downloadURL: 'https://r2cdn.beatsaver.com/test.zip',
      coverURL: 'https://cdn.beatsaver.com/test.jpg',
      diffs: [
        { characteristic: 'Standard', difficulty: 'Normal', notes: 3, nps: 0.3 },
        { characteristic: 'Standard', difficulty: 'Hard', notes: 3, nps: 0.3 },
      ],
    },
  ],
}
