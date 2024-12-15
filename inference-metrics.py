#!/usr/bin/env python3

"""
    $ inference-metrics.py [-n 5] -- ANY PROGRAM

このスクリプトは、`--` のあとに任意のプログラムを呼び出す文字列を受け取り、そのプログラムを `-n` で指定された回数（デフォルトは 5）だけ呼び出します。
"""

import re
import subprocess
import time

import click


def run_command(command) -> float:
    start_time = time.time()
    subprocess.run(command, check=True)
    return time.time() - start_time


def extract_metrics(stdout: str) -> dict:
    """Extract metrics from stdout"""
    metrics = {}

    # Extract prompt tokens and TTFT
    prompt_match = re.search(
        r"\[Prompt\]\s*=>\s*(\d+)\s*tokens,\s*latency\s*\(TTFT\):\s*([\d.]+)\s*ms",
        stdout,
    )
    if prompt_match:
        metrics["prompt_tokens"] = int(prompt_match.group(1))
        metrics["ttft_ms"] = float(prompt_match.group(2))

    # Extract extend tokens and TPS
    extend_match = re.search(
        r"\[Extend\]\s*=>\s*(\d+)\s*tokens,\s*throughput:\s*([\d.]+)\s*tokens/s", stdout
    )
    if extend_match:
        metrics["extend_tokens"] = int(extend_match.group(1))
        metrics["tps"] = float(extend_match.group(2))

    return metrics


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option("-n", default=5, help="Number of iterations (default: 5)")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def main(n: int, command):
    """Run a program multiple times and measure execution time"""
    if not command:
        raise click.UsageError("No command specified after '--'")

    for i in range(n):
        click.echo(f"\nRun {i + 1}/{n}")
        start_time = time.time()
        try:
            r = subprocess.run(command, check=True, text=True, capture_output=True)
            execution_time = time.time() - start_time
            stdout = r.stdout.strip()

            click.echo(f"Execution time: {execution_time:.3f}s")
            metrics = extract_metrics(stdout)
            if metrics:
                click.echo(f"Prompt tokens: {metrics.get('prompt_tokens', 'N/A')}")
                click.echo(f"TTFT: {metrics.get('ttft_ms', 'N/A')} ms")
                click.echo(f"Extend tokens: {metrics.get('extend_tokens', 'N/A')}")
                click.echo(f"Tokens/s: {metrics.get('tps', 'N/A')}")
            click.echo(f"stdout:\n{stdout}")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Command failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
