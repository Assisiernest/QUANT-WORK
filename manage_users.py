"""Manage Backtest Studio accounts from a terminal (also the way back in if you're locked out).

    python manage_users.py list
    python manage_users.py add alice --name "Alice Smith" [--admin] [--allow run_code local_files]
    python manage_users.py passwd alice          # prompts for the new password
    python manage_users.py unlock alice          # clear a lockout after too many wrong passwords
    python manage_users.py role alice admin|user
    python manage_users.py allow alice run_code local_files   (or: allow alice  -> no permissions)
    python manage_users.py delete alice
"""

from __future__ import annotations

import argparse
import getpass
import sys

from auth import PERMISSIONS, AuthError, UserStore


def ask_password() -> str:
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Again: "):
        raise AuthError("The two passwords don't match.")
    return first


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--name", default="")
    a.add_argument("--admin", action="store_true")
    a.add_argument("--allow", nargs="*", default=[], choices=PERMISSIONS)
    for cmd in ("passwd", "unlock", "delete"):
        sub.add_parser(cmd).add_argument("username")
    r = sub.add_parser("role")
    r.add_argument("username")
    r.add_argument("role", choices=["admin", "user"])
    al = sub.add_parser("allow")
    al.add_argument("username")
    al.add_argument("permissions", nargs="*", choices=PERMISSIONS)
    args = p.parse_args(argv)
    store = UserStore()
    try:
        if args.cmd == "list":
            users = store.list_users()
            if not users:
                print("No accounts yet. The first person to open the app creates the admin account.")
            for u in users:
                perms = "all" if u.is_admin else (", ".join(u.permissions) or "-")
                print(f"{u.username:20} {u.role:6} permissions: {perms:24} last login: {u.last_login[:16] or 'never'}")
        elif args.cmd == "add":
            store.create_user(args.username, ask_password(), args.name, "admin" if args.admin else "user", args.allow)
            print(f"Added {args.username}.")
        elif args.cmd == "passwd":
            store.set_password(args.username, ask_password())
            print(f"Password changed for {args.username}.")
        elif args.cmd == "unlock":
            store.unlock(args.username)
            print(f"Unlocked {args.username}.")
        elif args.cmd == "role":
            store.update(args.username, role=args.role)
            print(f"{args.username} is now {args.role}.")
        elif args.cmd == "allow":
            store.update(args.username, permissions=args.permissions)
            print(f"{args.username}: permissions {', '.join(args.permissions) or 'none'}.")
        elif args.cmd == "delete":
            store.delete_user(args.username)
            print(f"Deleted {args.username}. Their saved strategies and runs are still in app_data/users/.")
    except AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
