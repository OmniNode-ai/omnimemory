# OmniMemory Project Overview

## Purpose
Advanced unified memory and intelligence system designed to migrate and modernize 274+ intelligence modules from legacy omnibase_3 into a comprehensive, ONEX-compliant memory architecture. Accelerates development across all omni agents through systematic memory management, retrieval operations, and cross-modal intelligence patterns.

## Tech Stack
- **Python 3.12+** with Poetry dependency management
- **FastAPI + Uvicorn** for production API layer
- **Pydantic 2.10+** for data validation and ONEX compliance
- **Storage**: PostgreSQL/Supabase, ModelCachingSubcontract, Pinecone vector DB
- **ONEX Dependencies**: omnibase_spi, omnibase_core (git-based)
- **Async/await** architecture throughout

## Key Architecture Principles
- ONEX 4.0 compliance (zero `Any` types, strong typing)
- Protocol-based design patterns
- 4-node ONEX architecture (EFFECT → COMPUTE → REDUCER → ORCHESTRATOR)
- Async-first with comprehensive error handling
- Contract-driven development with Pydantic models

## Current Status
- Foundation models implemented (26 Pydantic models, zero Any types)  
- PR review phase with 10 remaining issues to resolve
- Directory structure: src/omnimemory/models/{core,memory,intelligence,service,foundation}