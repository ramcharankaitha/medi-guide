# MediGuide AI — Firestore Security Specification (Phase 0 Security TDD)

## 1. Data Invariants
1. **Master Gate & Ownership Invariant**: Every `/workspaces/{userId}` document and its subcollections (`/private/{docId}`, `/medicines/{itemId}`, `/reminders/{reminderId}`) can only be read, created, updated, or deleted when `request.auth != null`, `request.auth.token.email_verified == true`, `userId == request.auth.uid`, and `incoming().ownerId == request.auth.uid`.
2. **Subcollection Parent Existence Invariant**: No subcollection document (`/private/{docId}`, `/medicines/{itemId}`, `/reminders/{reminderId}`) can be accessed or mutated unless `get(/databases/$(database)/documents/workspaces/$(userId)).data.ownerId == request.auth.uid`.
3. **PII Split Collection Isolation**: Email and phone numbers are isolated inside `/workspaces/{userId}/private/{docId}` and never exposed to non-owners.
4. **Strict Key & Timestamp Enforcement**: Every `create` enforces `hasAll` and `hasOnly` on exact schema keys with `createdAt == request.time` and `updatedAt == request.time`. Every `update` wraps `isValid[Entity](incoming())`, locks `ownerId` and `createdAt`, enforces `updatedAt == request.time`, and restricts modified keys via `affectedKeys().hasOnly(...)`.

## 2. The "Dirty Dozen" Adversarial Payloads (All Must Return `PERMISSION_DENIED`)
1. **Payload 1 (Unverified Email Spoof)**: Authenticated user with `email_verified: false` attempting `create` on `/workspaces/uid1`.
2. **Payload 2 (Cross-User Workspace Read/PII Leak)**: User `uid2` attempting `get` on `/workspaces/uid1/private/info`.
3. **Payload 3 (Shadow Field Injection)**: `create` on `/workspaces/uid1` including unauthorized field `"isAdmin": true`.
4. **Payload 4 (OwnerId Spoofing on Create)**: `create` on `/workspaces/uid1/medicines/m1` where `ownerId: "victimUid"`.
5. **Payload 5 (OwnerId Mutation on Update)**: `update` on `/workspaces/uid1` changing `ownerId` from `"uid1"` to `"uid2"`.
6. **Payload 6 (Client Timestamp Forgery)**: `create` on `/workspaces/uid1/reminders/r1` with a past/future `createdAt` not equal to `request.time`.
7. **Payload 7 (CreatedAt Mutation)**: `update` on `/workspaces/uid1/medicines/m1` modifying `createdAt`.
8. **Payload 8 (ID Poisoning / Oversized Path ID)**: `create` on `/workspaces/uid1/medicines/{200_char_junk_id}`.
9. **Payload 9 (Denial-of-Wallet Array Overflow)**: `update` on `/workspaces/uid1` with `favouriteMedicineIds` array of 500 items (`> 50`).
10. **Payload 10 (Orphaned Subcollection Write)**: `create` on `/workspaces/uid1/medicines/m1` when parent `/workspaces/uid1` does not exist.
11. **Payload 11 (Terminal Feedback Mutation)**: `update` on `/feedback/fb1` when `resource.data.status == 'Resolved'` by a non-admin user.
12. **Payload 12 (Unbounded Blanket List Query)**: `list` on `/feedback` without filtering `resource.data.ownerId == request.auth.uid`.
