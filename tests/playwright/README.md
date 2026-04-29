# Playwright Tests for OptiMat Alloys

This directory contains Playwright-based browser automation tests that verify the complete new developer experience for OptiMat Alloys.

## Overview

The test suite simulates a new developer:
1. Cloning the GitHub repository
2. Starting Chainlit for the first time
3. Being prompted for an OpenAI API key
4. Entering the key and having it validated
5. Verifying the chat interface loads correctly
6. Testing a first interaction

## Prerequisites

### Install Playwright

```bash
# Install pytest-playwright
pip install pytest-playwright

# Install browser binaries
playwright install chromium
```

### Have a Valid OpenAI API Key

The tests read the API key from the development repo's `.env` file at:
```
/home/vladt/projects/OptiMat-Chat/.env
```

Make sure this file exists and contains a valid `OPENAI_API_KEY`.

## Running the Tests

### Run all Playwright tests:

```bash
pytest tests/playwright/ -v -s
```

### Run specific test file:

```bash
pytest tests/playwright/test_fresh_clone_experience.py -v -s
```

### Run with visible browser (headed mode):

```bash
pytest tests/playwright/ -v -s --headed
```

### Run with slower execution (useful for debugging):

```bash
pytest tests/playwright/ -v -s --headed --slowmo=1000
```

## What the Tests Do

### `test_fresh_clone_experience.py`

This comprehensive test simulates the complete first-time setup experience:

1. **Fresh Clone State** (`test_fresh_clone_state`)
   - Clones repo to `/tmp/optimat-test-{timestamp}/`
   - Verifies `.env` does NOT exist (clean slate)
   - Verifies `.env.example` exists
   - Verifies reference data files are present
   - Verifies public assets (favicon, logos) exist

2. **API Key Prompt** (`test_api_key_prompt_appears`)
   - Starts Chainlit in the cloned repo
   - Navigates browser to http://localhost:8002
   - Verifies API key prompt appears FIRST (blocking)
   - Takes screenshots of initial state

3. **API Key Submission** (`test_api_key_submission`)
   - Finds API key input field
   - Enters API key from dev `.env`
   - Clicks submit button
   - Waits for OpenAI API validation
   - Verifies chat interface appears after acceptance

4. **`.env` File Creation** (`test_env_file_created`)
   - Verifies `.env` file was created in cloned repo
   - Verifies file contains the API key
   - Verifies file permissions are 600 (Linux only)

5. **UI Assets Loading** (`test_ui_assets_load`)
   - Reloads page to check all asset requests
   - Verifies no 404 errors (favicon, logos, CSS, etc.)
   - Checks browser console for errors
   - Takes screenshot of loaded UI

6. **First Interaction** (`test_first_interaction`)
   - Types test message: "What is OptiMat Alloys?"
   - Submits message
   - Waits for agent response
   - Takes screenshot of interaction

## Test Output

### Console Output

Tests print detailed progress with ✓ marks for each step:

```
================================================================================
TEST 1: Verifying Fresh Clone State
================================================================================
✓ No .env file (expected)
✓ .env.example exists
✓ All 4 reference data files exist
✓ All 3 public assets exist
✓ TEST 1 PASSED: Fresh clone state verified
```

### Screenshots

All screenshots are saved to:
```
/tmp/optimat-test-{timestamp}/test_screenshots/
```

Screenshots captured:
- `01_initial_load.png` - First page load
- `02_api_key_prompt.png` - API key prompt modal
- `03_api_key_entered.png` - After entering key (before submit)
- `04_after_submission.png` - After clicking submit
- `05_assets_loaded.png` - Full page with assets loaded
- `06_first_interaction.png` - After first chat message

### Cleanup

The test automatically cleans up the temporary clone directory after completion.

## Troubleshooting

### Test fails with "No .env file found in dev repo"

Make sure `/home/vladt/projects/OptiMat-Chat/.env` exists and contains:
```
OPENAI_API_KEY=sk-...
```

### Test fails with "Chainlit server failed to start"

Possible issues:
- Port 8002 already in use → Kill existing process or change port in test
- Missing dependencies → Run `pip install -r requirements.txt`
- Conda environment not activated → Activate before running tests

### Test fails to find API key prompt

The test looks for common indicators like "api key", "openai", etc. If the prompt text changed:
1. Check screenshot: `01_initial_load.png`
2. Update the `api_key_indicators` list in the test
3. Or update the prompt text in `run_chat.py` to match

### Test fails to find input field or submit button

The test tries multiple common selectors. If the UI changed:
1. Check screenshot: `02_api_key_prompt.png`
2. Inspect the actual HTML selectors in the Chainlit UI
3. Update `input_selectors` or `submit_selectors` in the test

### Browser doesn't appear (headless mode)

By default, tests run in headless mode. To see the browser:
```bash
pytest tests/playwright/ -v -s --headed
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Playwright tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    playwright install chromium
    pytest tests/playwright/ -v
```

**Note:** You'll need to:
1. Store `OPENAI_API_KEY` as a GitHub secret
2. Modify the test to read from environment variable instead of `.env` file in CI

## Test Philosophy

These tests are **integration tests** that verify the complete user experience, not unit tests. They:
- Test the real application (not mocked)
- Use real API calls (small cost, ~$0.001 per test run)
- Verify the actual UI (not just backend logic)
- Catch issues that unit tests miss (CSS, assets, timing, etc.)

## Future Enhancements

Potential additions to the test suite:

1. **Test invalid API keys**: Wrong format, revoked keys, expired keys
2. **Test subsequent launches**: Restart Chainlit, verify no prompt appears
3. **Test tool execution**: Generate a structure, verify images appear
4. **Test calculator selection**: Change calculator in settings, verify it works
5. **Test error scenarios**: Network failures, permission errors, etc.

## Contributing

When adding new features to OptiMat Alloys, please:
1. Update existing tests if UI changes
2. Add new tests for new user-facing features
3. Run tests before committing: `pytest tests/playwright/ -v`
4. Include screenshots in PR if UI changed significantly
