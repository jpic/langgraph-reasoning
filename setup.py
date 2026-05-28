from setuptools import setup, find_packages

setup(
    name="langchain-reasoning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-openai>=0.2.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
            "langchain-deepseek>=0.1.0",
        ]
    },
)
