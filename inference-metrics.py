#!/usr/bin/env python3

"""
    $ inference-metrics.py [-n 5] -- ANY PROGRAM

このスクリプトは、`--` のあとに任意のプログラムを呼び出す文字列を受け取り、そのプログラムを `-n` で指定された回数（デフォルトは 5）だけ呼び出します。
"""

import subprocess
import time

import click


def run_command(command) -> float:
    start_time = time.time()
    subprocess.run(command, check=True)
    return time.time() - start_time


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
        try:
            r = subprocess.run(command, check=True, text=True)
            click.echo(f"stdout:\n{r.stdout}")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Command failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
