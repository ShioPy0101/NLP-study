import argparse
import json
import urllib.request


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--min-len-before-eos", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "min_len_before_eos": args.min_len_before_eos,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        args.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    print(result["text"])


if __name__ == "__main__":
    main()
