import sys

USAGE = """usage:
  python -m dnn node [options]   run a node on this device (needs Python 3 + NumPy)
  python -m dnn run  [options]   run the coordinator (measures, plans, deploys, chats)

  python -m dnn.node --help / python -m dnn.run --help for all options."""


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("node", "run"):
        print(USAGE)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "node":
        from .node import main as m
    else:
        from .run import main as m
    m(rest)


if __name__ == "__main__":
    main()
