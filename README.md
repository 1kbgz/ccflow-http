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

- Implemented: `HTTPConfig`, `HTTPAuth`, `HTTPContext`, `HTTPRequestContext`, `HTTPRequest`, `HTTPRetryPolicy`, `HTTPResponseResult`, compatibility `HTTPResult`, `HTTPModel`, templated path/query/header rendering, request explanation through `build_request`, no-auth/bearer/API-key/basic auth helpers, JSON/text/bytes/CSV/gzip response parsing, HTTP status retry classification over `ccflow` retry semantics, retry event summaries on HTTP results, `ccflow-etl` `ExecutionPolicy` request spacing, `next_url`/cursor/page/offset pagination, rate-limit header capture, and mocked `httpx2` transport tests.
- Partial: the shared execution policy is currently consumed inside `HTTPModel` for sequential request spacing; broader evaluator-level concurrency coordination still belongs in evaluator integrations.
- Missing: broader integration examples and provider-specific subclasses/configs in downstream packages.

## Dependency Contract

- Depends on `ccflow` for callable model, context, and result interfaces.
- Depends on `ccflow` retry policy semantics and `ccflow-etl` execution policy models.
- Must not depend on finance packages or application-specific packages.

## Test Convention

Default tests should use mocked `httpx2` transports or local fixtures. They should not require live network calls or provider credentials.

> [!NOTE]
> This library was generated using [copier](https://copier.readthedocs.io/en/stable/) from the [Base Python Project Template repository](https://github.com/python-project-templates/base).
