#!/bin/bash
# validate-no-fallbacks.sh - Pre-commit hook to detect and prevent fallback patterns

set -e

echo "🔍 Validating no fallback patterns are present..."

# Define fallback patterns that should not exist in production code
FALLBACK_PATTERNS=(
    "except ImportError:"
    "# Fallback for development"
    "# Mock.*for development"
    "if.*not.*_AVAILABLE"
    "class Mock"
    "# Fallback"
    "try:.*except.*ImportError"
    "AVAILABLE = False"
)

# Files to check (exclude test files, scripts, and development utilities)
FILES_TO_CHECK=$(find src/ -name "*.py" -not -path "*/test*" -not -path "*/scripts/*" -not -path "*/__pycache__/*")

FALLBACKS_FOUND=0
FAILED_FILES=()

echo "📁 Checking files for fallback patterns..."

for pattern in "${FALLBACK_PATTERNS[@]}"; do
    echo "  🔎 Checking pattern: $pattern"

    while IFS= read -r file; do
        if grep -l "$pattern" "$file" 2>/dev/null; then
            echo "    ❌ FALLBACK DETECTED in $file:"
            grep -n "$pattern" "$file" | sed 's/^/      /'
            FALLBACKS_FOUND=$((FALLBACKS_FOUND + 1))
            FAILED_FILES+=("$file")
        fi
    done <<< "$FILES_TO_CHECK"
done

# Check for specific problematic fallback classes
echo "  🔎 Checking for mock classes..."
if grep -r "class.*Client:" src/ --include="*.py" | grep -v test | grep -E "(Mock|Fake)"; then
    echo "    ❌ MOCK CLIENT CLASSES DETECTED:"
    grep -r "class.*Client:" src/ --include="*.py" | grep -v test | grep -E "(Mock|Fake)" | sed 's/^/      /'
    FALLBACKS_FOUND=$((FALLBACKS_FOUND + 1))
fi

# Check for availability flags
echo "  🔎 Checking for availability flags..."
if grep -r "_AVAILABLE.*=" src/ --include="*.py" | grep -v test; then
    echo "    ❌ AVAILABILITY FLAGS DETECTED:"
    grep -r "_AVAILABLE.*=" src/ --include="*.py" | grep -v test | sed 's/^/      /'
    FALLBACKS_FOUND=$((FALLBACKS_FOUND + 1))
fi

# Summary
if [ $FALLBACKS_FOUND -gt 0 ]; then
    echo ""
    echo "🚨 VALIDATION FAILED: $FALLBACKS_FOUND fallback pattern(s) detected"
    echo ""
    echo "📋 Failed files:"
    for file in "${FAILED_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "💡 Fallbacks mask real dependency issues and create false test confidence."
    echo "   Remove fallbacks and fix the underlying dependency resolution problems."
    echo ""
    exit 1
else
    echo "✅ No fallback patterns detected - all dependencies must be properly resolved"
fi