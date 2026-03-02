#!/bin/bash
set -e
mkdir -p /home/claude/.claude
chown -R claude:claude /home/claude/.claude
exec gosu claude "$@"
