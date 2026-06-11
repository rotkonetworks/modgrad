#!/usr/bin/env bash
# Build the OCR Online static site and deploy to the two dockers LXCs
# behind the rotko anycast haproxy.
#
# Pre-requisites (one-time):
#   - SSH access to bkk07 and bkk08 (via ~/.ssh/config aliases).
#   - `nginx` server block `ocr.rotko.net` installed inside each
#     dockers LXC (CT 992 on bkk07, CT 982 on bkk08), root
#     `/srv/ocr-site`. See `deploy/nginx-ocr.conf`.
#   - haproxy patched with `ocr-backend` on bkk06/07/08 host AND
#     LXC haproxies. See `deploy/haproxy-ocr.patch`.
#   - DNS: `ocr.rotko.net  A  160.22.181.81`
#   - TLS: cert provisioned via `certs.sh ocr.rotko.net` on any
#     haproxy LXC after DNS resolves.
#
# Run from anywhere; the script changes into the project root itself.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "==> Type-checking"
npm run typecheck

echo "==> Building"
npm run build

echo "==> Packing dist"
TARBALL=/tmp/ocr-dist-$$.tar.gz
tar -C dist -czf "$TARBALL" .
trap 'rm -f "$TARBALL"' EXIT
echo "    $TARBALL ($(du -h "$TARBALL" | cut -f1))"

declare -a TARGETS=("bkk07:992" "bkk08:982")
for target in "${TARGETS[@]}"; do
  host="${target%:*}"
  ct="${target#*:}"
  echo "==> Deploying to $host CT $ct"
  scp -q "$TARBALL" "$host:/tmp/ocr-dist.tar.gz"
  ssh "$host" "pct push $ct /tmp/ocr-dist.tar.gz /tmp/ocr-dist.tar.gz"
  ssh "$host" "pct exec $ct -- bash -c '
    set -e
    cd /srv/ocr-site
    rm -rf assets content.json embed* index.html manifest.webmanifest og.svg robots.txt sitemap.xml sw.js
    tar xzf /tmp/ocr-dist.tar.gz
    ls
  '" | sed 's/^/    /'
done

echo "==> Sanity check via anycast"
ssh bkk07 'curl -k -fsS --resolve ocr.rotko.net:443:160.22.181.81 -H "Host: ocr.rotko.net" https://ocr.rotko.net/__health' \
  | sed 's/^/    /'
ssh bkk07 'curl -k -fsS --resolve ocr.rotko.net:443:160.22.181.81 -H "Host: ocr.rotko.net" -o /dev/null -w "    /        %{http_code}\\n" https://ocr.rotko.net/'
ssh bkk07 'curl -k -fsS --resolve ocr.rotko.net:443:160.22.181.81 -H "Host: ocr.rotko.net" -o /dev/null -w "    /embed.html         %{http_code}\\n" https://ocr.rotko.net/embed.html'
ssh bkk07 'curl -k -fsS --resolve ocr.rotko.net:443:160.22.181.81 -H "Host: ocr.rotko.net" -o /dev/null -w "    /embed-client.js    %{http_code}\\n" https://ocr.rotko.net/embed-client.js'
ssh bkk07 'curl -k -fsS --resolve ocr.rotko.net:443:160.22.181.81 -H "Host: ocr.rotko.net" -o /dev/null -w "    /content.json       %{http_code}\\n" https://ocr.rotko.net/content.json'

echo "==> Done"
