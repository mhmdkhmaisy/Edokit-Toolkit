#!/bin/bash
set -e

# Resolve/refresh Maven dependencies for the edokit Java project so a fresh
# merge always has a warm local repository before the next build.
cd "$(dirname "$0")/../edokit"
mvn -q -DskipTests dependency:resolve
