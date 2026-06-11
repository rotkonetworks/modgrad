# cdn.rotko.net — model weight hosting

MinIO behind Caddy. Two-node bucket replication between bkk07 and bkk08.

## DNS

    cdn.rotko.net           A    <bkk07 ip>
    cdn.rotko.net           A    <bkk08 ip>
    admin.cdn.rotko.net     A    <bkk07 ip>   # console, restricted

Round-robin failover. Cloudflare proxy with health checks works too.

## Per-node bring-up

On both bkk07 and bkk08:

    cp .env.example .env && $EDITOR .env
    docker compose up -d
    docker compose logs -f caddy   # wait for TLS cert
    mc alias set local https://cdn.rotko.net \
        "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
    mc mb local/ocr
    mc anonymous set download local/ocr

## Replication (run once, from either node)

    mc alias set bkk07 https://cdn-bkk07.rotko.net "$USER" "$PASS"
    mc alias set bkk08 https://cdn-bkk08.rotko.net "$USER" "$PASS"
    mc admin replicate add bkk07 bkk08

`mc admin replicate info bkk07` confirms the link. Writes on either
node propagate to the other; reads served by whichever DNS resolves to.

## Publishing weights

    mc cp eight_region_small.bin local/ocr/v1/

Resulting URL the browser fetches:

    https://cdn.rotko.net/ocr/v1/eight_region_small.bin

Matches the default in the demo's `src/state/model.ts`.

## Notes

- Caddy auto-provisions Let's Encrypt certs on first request; DNS must
  point at the host before that succeeds.
- The public face is GET/HEAD/OPTIONS only. PUT/POST/DELETE come
  through the admin subdomain.
- `Cache-Control: immutable` means versioned URLs are cached forever
  client-side. Bump the path version (`/ocr/v2/…`) to push an update.
