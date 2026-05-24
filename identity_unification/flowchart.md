# Identity Unification — Flowchart

```mermaid
flowchart TD
    A[Raw contact ingested<br/>email / phone / handle / display name] --> B[Normalize<br/>lowercase, strip, split email local-part,<br/>E.164 phone, strip @ from handles]
    B --> C{Deterministic key match?<br/>email or phone hits an existing profile}
    C -- yes --> D[Auto-link to that profile<br/>confidence = 1.00]
    C -- no  --> E[Generate candidate links<br/>across all profiles]

    E --> F[Score each candidate<br/>weighted blend:<br/>0.45 * email-local fuzz<br/>+ 0.35 * name fuzz<br/>+ 0.20 * handle fuzz<br/>+ phone-suffix boost]
    F --> G{Best score?}

    G -- ">= 0.80" --> H[Auto-link bucket]
    G -- "0.55–0.80" --> I[Verify-manually bucket<br/>push to human queue]
    G -- "< 0.55"   --> J[Reject — likely a new profile<br/>create canonical record]

    H --> K[(Unified profile store)]
    D --> K
    I --> L[(Human review queue)]
    J --> K
    L --> K
```

## Key ideas

- **Deterministic before probabilistic.** Exact email/phone hits skip
  the fuzzy step entirely — those are essentially primary keys.
- **Weighted blend of fuzzy signals.** Email local-part is the strongest
  on-platform name proxy, display name is next, social handle last.
  A matching phone-number suffix adds a small bonus.
- **Three buckets**: auto / verify / reject. The mid bucket is where
  human-in-the-loop earns its keep — we never auto-merge ambiguous
  identities.
- **Output**: every linked contact ends up in a canonical
  `unified_profile`, but ambiguous ones go through the review queue first.
