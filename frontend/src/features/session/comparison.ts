/*
 * The controlled fly against its untouched twin (PROTOCOL.md section 3: slot 1 plays the
 * same brain, song and seed with no controls).
 */

export interface Comparison {
  /** Hit rates, 0 to 1. */
  fly: number
  twin: number
  /** fly minus twin, in percentage points. */
  points: number
  /** Notes each has judged; the comparison means little over a handful. */
  judged: number
}

/** The controlled fly against its untouched twin, once both have judged a note. */
export function compareScores(
  score: { hits: number; misses: number; accuracy: number } | null,
  twin: { hits: number; misses: number; accuracy: number } | null,
): Comparison | null {
  if (!score || !twin) return null
  const judged = Math.min(score.hits + score.misses, twin.hits + twin.misses)
  if (judged === 0) return null
  return {
    fly: score.accuracy,
    twin: twin.accuracy,
    points: Math.round((score.accuracy - twin.accuracy) * 1000) / 10,
    judged,
  }
}
