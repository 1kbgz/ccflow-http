# ccflow-http

ccflow models for HTTP

[![Build Status](https://github.com/1kbgz/ccflow-http/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/1kbgz/ccflow-http/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/1kbgz/ccflow-http/branch/main/graph/badge.svg)](https://codecov.io/gh/1kbgz/ccflow-http)
[![License](https://img.shields.io/github/license/1kbgz/ccflow-http)](https://github.com/1kbgz/ccflow-http)
[![PyPI](https://img.shields.io/pypi/v/ccflow-http.svg)](https://pypi.python.org/pypi/ccflow-http)

## Overview

`ccflow-http` provides public, domain-neutral HTTP callable models for `ccflow` workflows. It should own request configuration, auth strategies, request templating, pagination, response parsing, timeout handling, retry/rate-limit integration, and HTTP result metadata.

It should not contain provider-specific endpoint catalogs. Domain packages can configure or subclass these generic models for particular APIs.

## Current Status

- Implemented: `HTTPConfig`, `HTTPAuth`, `HTTPContext`, `HTTPRequestContext`, `HTTPRequest`, `HTTPResponseResult`, compatibility `HTTPResult`, `HTTPModel`, templated path/query/header rendering, request explanation through `build_request`, no-auth/bearer/API-key/basic auth helpers, JSON/text/bytes/CSV/gzip response parsing, `ccflow-etl` `RetryPolicy` integration for retry classification, `next_url`/cursor/page/offset pagination, rate-limit header capture, and mocked `httpx` transport tests.
- Partial: retry behavior consumes shared retry policy models, but backoff/jitter scheduling and retry event summaries are not yet implemented.
- Missing: broader integration examples and provider-specific subclasses/configs in downstream packages.

## Dependency Contract

- Depends on `ccflow` for callable model, context, and result interfaces.
- Depends on transport-neutral retry policy models from `ccflow-etl`.
- Must not depend on finance packages or application-specific packages.

## Test Convention

Default tests should use mocked `httpx` transports or local fixtures. They should not require live network calls or provider credentials.

> [!NOTE]
> This library was generated using [copier](https://copier.readthedocs.io/en/stable/) from the [Base Python Project Template repository](https://github.com/python-project-templates/base).
