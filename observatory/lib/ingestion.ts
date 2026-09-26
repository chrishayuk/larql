/** Transport boundary: source bytes → validated run → existing views. */
import { openRecording } from './standard.ts';
import { sourceDigest } from './lenses.ts';
import type { LensRecord } from './lenses.ts';
import { readVindex3Record } from './vindex3-record.ts';

export async function openRecordingText(bytes: string) {
  if (new TextEncoder().encode(bytes).length > 10_000_000) throw new Error('Recording exceeds the 10 MB import limit.');
  // Detect the framing, never catch a failed V3 validation and fall back to fixtures.
  const first = bytes.split(/\r?\n/).find(line => line.trim());
  let envelope: unknown;
  try { envelope = JSON.parse(first ?? ''); } catch { /* Pretty-printed legacy JSON. */ }
  if (envelope && typeof envelope === 'object' && 'header' in envelope) return readVindex3Record(bytes);
  const opened = openRecording(JSON.parse(bytes));
  return { ...opened, bytes, digest: await sourceDigest(bytes), format: 'json' as const, lenses: undefined as LensRecord | undefined };
}
