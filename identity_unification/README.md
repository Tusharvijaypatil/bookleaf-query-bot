# Identity Unification (Intermediate Task)

Links one author's identities across email, WhatsApp/phone, dashboard
display name and Instagram handle into a single canonical profile,
with a confidence score and a manual-verify fallback.

## Run it

From the repo root:

```bash
python -m identity_unification.unify
```

You'll get a `rich` table of every candidate cross-platform pair,
ranked by confidence, coloured by bucket.

## Approach

1. **Normalize** every contact:
   - email → lowercase, take the local-part, strip non-alphanumerics
   - phone / WhatsApp → digits-only, keep the last 7 as `phone_suffix`
   - Instagram → strip the leading `@`
   - display name → lowercase, strip punctuation

2. **Score every cross-platform pair**:
   - If both sides have a phone and the last 7 digits match, short-circuit
     to **0.95** — same phone is effectively a primary-key match. (We
     stop short of 1.0 so a future name-discrepancy check can still
     veto.)
   - Otherwise, weighted blend of `rapidfuzz.WRatio` over name-ish fields:

   ```
   score = 0.45 * email_local_signal
         + 0.35 * name_signal
         + 0.20 * handle_signal
   ```

   Same-platform pairs are skipped — we never collapse two emails into
   one identity by name similarity alone.

3. **Bucket** by threshold:

   | Score      | Bucket             | What happens                       |
   |------------|--------------------|------------------------------------|
   | `>= 0.80`  | `auto_link`        | merged into the canonical profile  |
   | `0.55–0.80`| `verify_manually`  | pushed to the human review queue   |
   | `< 0.55`   | `reject`           | treated as a separate identity     |

## What I'd improve with more time

- **Deterministic linking keys.** Verified email or phone → instant link
  with confidence 1.0, no fuzzy step needed. The current demo treats
  all signals probabilistically; in practice, a confirmed email or
  phone is an effectively unique key.
- **Embedding-based name matching.** `rapidfuzz` handles typos and
  abbreviations but breaks on transliteration (`Sara` vs `Saara`) and
  nickname maps (`Priya` vs `P.`). A small embedding model over names
  would handle these gracefully.
- **Graph clustering instead of pairwise scoring.** Build a graph of
  contacts with edges weighted by score, then run connected-components
  with a confidence floor. This handles transitive links (A↔B↔C all
  belong together even if A–C scores below threshold).
- **Active human-in-the-loop queue.** The `verify_manually` bucket
  should land in a real review UI with one-click merge/reject, plus
  feedback flowing back to tune the weights.
- **Per-platform handle conventions.** `@sarapoetry23` and
  `sara_j_dash` aren't directly comparable — splitting on common
  delimiters and matching token-by-token improves recall.
- **Drift monitoring.** Periodically resample the auto-linked bucket
  for human spot-check to make sure precision hasn't decayed.
