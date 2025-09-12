#!/usr/bin/env python3
import argparse, bcrypt, getpass, sys

def main():
    ap = argparse.ArgumentParser(description="Generate bcrypt hash for ADMIN_PASSWORD_HASH")
    ap.add_argument("-r", "--rounds", type=int, default=12, help="bcrypt cost (default: 12)")
    args = ap.parse_args()

    pw1 = getpass.getpass("New admin password: ")
    pw2 = getpass.getpass("Confirm: ")
    if pw1 != pw2:
        print("Error: passwords do not match.", file=sys.stderr)
        sys.exit(1)

    h = bcrypt.hashpw(pw1.encode(), bcrypt.gensalt(rounds=args.rounds)).decode()
    print(h)

if __name__ == "__main__":
    main()
