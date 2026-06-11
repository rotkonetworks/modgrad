#!/usr/bin/env python3
"""Apply the ocr.rotko.net haproxy patch in-place on /etc/haproxy/haproxy.cfg.

Idempotent: if the patch is already applied, this script no-ops and
exits 0. Validates with `haproxy -c -f` before swapping in the new file.

Usage:
    scp this file to the target node, run as root:
        sudo python3 apply-haproxy.py

Run on every anycast peer: bkk06, bkk07, bkk08 hosts AND the haproxy
LXCs on each of those hosts (CT 9916, 991, 989).
"""

from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

CFG = Path('/etc/haproxy/haproxy.cfg')

OCR_ACL = '    acl is_ocr hdr_end(host) -i ocr.rotko.net'
OCR_USE_BACKEND = '    use_backend ocr-backend if is_ocr'
OCR_BACKEND_BLOCK = '''
# OCR Online (ocr.rotko.net)
backend ocr-backend
    mode http
    balance leastconn
    http-request set-var(txn.skip_xfo) bool(true)
    option httpchk
    http-check send meth GET uri /__health hdr Host ocr.rotko.net
    http-check expect status 200
    http-response set-header X-Served-By "ocr-rotko-net"
    server bkk07-ocr 192.168.77.92:80 check inter 5s fall 3 rise 2
    server bkk08-ocr 192.168.78.92:80 check inter 5s fall 3 rise 2
'''

ASTROLABE_BACKEND_END = (
    '# Astrolabe Backend (astrolabe.rotko.net)\n'
    'backend astrolabe-backend\n'
    '    mode http\n'
    '    server docker-astrolabe 192.168.77.97:80 check inter 2s'
)


def main() -> int:
    src = CFG.read_text()
    if 'txn.skip_xfo' in src and 'ocr-backend' in src:
        print('already patched')
        return 0

    out = src
    if OCR_ACL not in out:
        out = out.replace(
            '    acl is_astrolabe hdr_end(host) -i astrolabe.rotko.net',
            '    acl is_astrolabe hdr_end(host) -i astrolabe.rotko.net\n' + OCR_ACL,
            1,
        )
    if OCR_USE_BACKEND not in out:
        out = out.replace(
            '    use_backend astrolabe-backend if is_astrolabe',
            '    use_backend astrolabe-backend if is_astrolabe\n' + OCR_USE_BACKEND,
            1,
        )
    if 'backend ocr-backend' not in out:
        if ASTROLABE_BACKEND_END not in out:
            print('error: astrolabe backend block not found; cannot place ocr-backend')
            return 1
        out = out.replace(
            ASTROLABE_BACKEND_END,
            ASTROLABE_BACKEND_END + '\n' + OCR_BACKEND_BLOCK,
            1,
        )
    # Make the global X-Frame-Options conditional so the embed iframe
    # can be loaded from any origin.
    old_xfo = 'http-response set-header X-Frame-Options "DENY"'
    new_xfo = 'http-response set-header X-Frame-Options "DENY" if !{ var(txn.skip_xfo) -m bool }'
    if old_xfo in out and new_xfo not in out:
        out = out.replace(old_xfo, new_xfo, 1)

    backup = CFG.with_suffix(f'.cfg.bak-{int(time.time())}')
    backup.write_text(src)
    staged = CFG.with_suffix('.cfg.new')
    staged.write_text(out)

    print(f'wrote {staged} (backup: {backup})')
    r = subprocess.run(['haproxy', '-c', '-f', str(staged)], capture_output=True, text=True)
    if r.returncode != 0:
        print('haproxy -c rejected the new config:')
        print(r.stdout)
        print(r.stderr)
        return r.returncode
    print('haproxy -c OK')

    staged.rename(CFG)
    r = subprocess.run(['systemctl', 'reload', 'haproxy'])
    if r.returncode != 0:
        print('reload failed')
        return r.returncode
    print('reloaded')
    return 0


if __name__ == '__main__':
    sys.exit(main())
