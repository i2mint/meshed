from setuptools import setup


def text_of_readme_md_file():
    try:
        with open('README.md') as f:
            return f.read()
    except:
        return ""


setup(
    long_description=text_of_readme_md_file(),
    long_description_content_type="text/markdown"
)  # Note: Everything should be in the local setup.cfg
