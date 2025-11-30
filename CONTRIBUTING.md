# Contributing to BYU Pathway Topic Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

Before contributing, please:

1. **Read the [Wiki](https://github.com/chrisedeson/pathway-questions-topic-modelling/wiki)** to understand the system
2. **Contact Christopher or Ernest** for access to:
   - Google Colab notebook
   - AWS credentials
   - Topics Google Sheet
   - Streamlit dashboard (if deployed)

## Types of Contributions

We welcome contributions in these areas:

### 1. Dashboard Improvements
- New visualizations or charts
- Additional analytics pages
- UI/UX enhancements
- Performance optimizations

### 2. Notebook Enhancements
- Better clustering algorithms
- Improved topic matching
- Additional data processing features
- Performance optimizations

### 3. Documentation
- Wiki updates
- Code comments
- Usage examples
- Troubleshooting guides

### 4. Bug Fixes
- Fix reported issues
- Improve error handling
- Resolve edge cases

### 5. Testing
- Add test cases
- Improve test coverage
- Integration testing

## Development Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Git
git --version
```

### Clone Repository

```bash
git clone https://github.com/chrisedeson/pathway-questions-topic-modelling.git
cd pathway-questions-topic-modelling
```

### Install Dependencies

```bash
# For dashboard development
cd v2.0.0
pip install -r requirements.txt

# For notebook development
cd ../notebook
pip install -r requirements.txt  # if requirements.txt exists
```

### Configure Environment

Create `.env` file in project root:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
OPENAI_API_KEY=your_openai_key
```

**Important:** Never commit `.env` to git! It's in `.gitignore` by default.

### Run Locally

```bash
# Start dashboard
cd v2.0.0
streamlit run app.py
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation if needed

### 3. Test Your Changes

#### Dashboard Changes
```bash
# Run locally and test:
# - All pages load
# - Filters work
# - Charts render
# - Export works
# - No console errors
```

#### Notebook Changes
```bash
# Test in Google Colab:
# - Sample mode runs successfully
# - Output files generated
# - S3 upload works
# - Error log is reasonable
```

### 4. Commit Changes

```bash
git add .
git commit -m "Brief description of changes"
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Be concise but descriptive
- Reference issue numbers if applicable

**Examples:**
```bash
git commit -m "Add sentiment analysis to dashboard"
git commit -m "Fix timezone display issue in charts"
git commit -m "Update Quick Reference with new commands"
git commit -m "Improve clustering performance by 30%"
```

### 5. Push to GitHub

```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request

1. Go to [GitHub repository](https://github.com/chrisedeson/pathway-questions-topic-modelling)
2. Click "Pull Requests" → "New Pull Request"
3. Select your branch
4. Fill in PR template:
   - **Title:** Clear, concise description
   - **Description:** What changed and why
   - **Testing:** How you tested the changes
   - **Screenshots:** If UI changed

**PR Example:**
```markdown
## Summary
Add regional sentiment analysis to Regional Insights page

## Changes
- Added sentiment classification by country
- Created new visualization for sentiment distribution
- Added filtering by sentiment category

## Testing
- Tested with sample data (1000 questions)
- Verified all countries show sentiment breakdown
- Tested filters work correctly
- Checked no performance regression

## Screenshots
[Attach screenshot of new feature]
```

## Code Style

### Python
- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable names
- Add docstrings for functions
- Keep functions focused and small

**Example:**
```python
def calculate_similarity_score(question1: str, question2: str) -> float:
    """
    Calculate cosine similarity between two questions.

    Args:
        question1: First question text
        question2: Second question text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Implementation here
    pass
```

### Streamlit
- One page per file in `pages/` directory
- Clear page titles with emojis
- Use Streamlit caching for expensive operations
- Handle loading states gracefully

**Example:**
```python
import streamlit as st
import pandas as pd

st.title("📊 My New Page")

@st.cache_data(ttl=3600)
def load_data():
    """Load data with caching"""
    # Load logic here
    return data

data = load_data()
st.dataframe(data)
```

### Configuration
- Put configurable values in `config.py`
- Use environment variables for secrets
- Document all configuration options

## Testing Guidelines

### Before Submitting PR

**Dashboard changes:**
- [ ] All pages load without errors
- [ ] Filters work correctly
- [ ] Charts render properly
- [ ] Export to CSV works
- [ ] Dark/light theme both work
- [ ] No browser console errors
- [ ] Tested with different data sizes

**Notebook changes:**
- [ ] Sample mode runs successfully
- [ ] Full mode tested (if applicable)
- [ ] Output files generated correctly
- [ ] S3 upload works
- [ ] Error log reviewed
- [ ] No API errors
- [ ] Performance acceptable

**Documentation changes:**
- [ ] Links work
- [ ] Markdown renders correctly
- [ ] Code examples tested
- [ ] No typos or grammar issues

## Adding New Features

### Dashboard Pages

1. Create new file in `v2.0.0/pages/`:
   ```python
   # pages/7_📊_My_New_Page.py
   import streamlit as st

   st.title("📊 My New Page")
   # Your code here
   ```

2. Use utility functions from `utils/`:
   - `data_loader.py` - Load data from S3
   - `visualizations.py` - Chart helpers

3. Test thoroughly before submitting

### Notebook Features

1. **Discuss with team first** - Notebook changes affect data pipeline
2. Test in Google Colab thoroughly
3. Document changes in notebook markdown cells
4. Update repository reference notebook

### Analytics Features

1. Add to appropriate analytics tab in `pages/3_📈_Trends_&_Analytics.py`
2. Use BYU colors from `config.py`
3. Make visualizations interactive (hover, zoom, etc.)
4. Add helpful tooltips and labels

## Documentation

### When to Update Docs

- **Always** update docs when changing functionality
- Add to wiki for user-facing changes
- Add code comments for implementation details

### Wiki Updates

To update wiki:

1. Clone wiki repository:
   ```bash
   git clone https://github.com/chrisedeson/pathway-questions-topic-modelling.wiki.git
   ```

2. Edit markdown files

3. Commit and push:
   ```bash
   git add .
   git commit -m "Update documentation for X feature"
   git push
   ```

### README Updates

- Keep README.md minimal
- Point to wiki for details
- Only update for major changes

## Versioning

This project uses semantic versioning:

- **Major** (v3.0.0): Breaking changes, major rewrites
- **Minor** (v2.1.0): New features, non-breaking changes
- **Patch** (v2.0.1): Bug fixes, small improvements

Current version: **v2.0.0**

## Review Process

1. **Submit PR** with clear description
2. **Automated checks** run (if configured)
3. **Team review** - Christopher or Ernest reviews
4. **Address feedback** if requested
5. **Merge** once approved

## Questions?

- **Technical questions:** Contact Christopher or Ernest
- **Bug reports:** Open an [issue](https://github.com/chrisedeson/pathway-questions-topic-modelling/issues)
- **Feature requests:** Open an [issue](https://github.com/chrisedeson/pathway-questions-topic-modelling/issues)
- **Documentation:** See the [Wiki](https://github.com/chrisedeson/pathway-questions-topic-modelling/wiki)

## Code of Conduct

- Be respectful and professional
- Help others learn and grow
- Accept constructive feedback gracefully
- Focus on what's best for the project

## License

By contributing, you agree that your contributions will be part of this BYU Pathway Worldwide project.

---

Thank you for contributing to BYU Pathway Topic Analysis! Your contributions help improve student support services.
