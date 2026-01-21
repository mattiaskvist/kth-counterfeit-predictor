from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")


class CounterfeitPrediction(BaseModel):
    """Structured output for counterfeit prediction."""

    is_counterfeit: bool = Field(description="True if the product is likely counterfeit, False if genuine")


class CounterfeitPredictionMachine:
    """
    Detects counterfeit products using LLM-based classification.

    The Counterfeit machine predicts whether a product is likely counterfeit or genuine.
    This is done one product at a time.

    NOTE:
    1. Currently this is a basic implementation using few-shot prompting. Change this to your own approach!
    2. You are allowed to change a lot here! But the prediction function should handle a single product at a time for the tests to work and the submit results script to work.
    3. Can you think of a way to use the training data to improve the prediction?
    4. You are not allowed to train on the eval or test data, obviously!
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        )
        self.chain = None
        self.few_shot_examples = []

        self.train_df = pd.read_csv(PROJECT_ROOT / "data" / "products_train.csv")

    def _format_product(self, product: pd.Series) -> str:
        """Format a product row as a string for the LLM."""
        return f"Product: {json.dumps(product.to_dict())}"

    def _train(self) -> None:
        """
        Train the model using the provided training data.

        NOTE: This is currently a basic implementation using few-shot prompting.

        Args:
            train_df: DataFrame with product features and 'label' column
        """
        # Select a few examples of each class for few-shot learning
        genuine_examples = self.train_df[self.train_df["label"] == 0].head(3)
        counterfeit_examples = self.train_df[self.train_df["label"] == 1].head(3)

        # Build few-shot examples
        self.few_shot_examples = []
        for _, row in genuine_examples.iterrows():
            self.few_shot_examples.append(
                {
                    "input": self._format_product(row),
                    "output": "genuine",
                }
            )
        for _, row in counterfeit_examples.iterrows():
            self.few_shot_examples.append(
                {
                    "input": self._format_product(row),
                    "output": "counterfeit",
                }
            )

        # Create the few-shot prompt template
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Analyze this product listing:\n{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.few_shot_examples,
        )

        # Create the full prompt with system message and few-shot examples
        full_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are an expert at detecting counterfeit products in e-commerce listings.
            Analyze the product features and determine if the listing is likely GENUINE or COUNTERFEIT.
            Respond with your classification.
            """,
                ),
                few_shot_prompt,
                ("human", "Analyze this product listing:\n{input}"),
            ]
        )

        # Create the chain with structured output
        self.chain = full_prompt | self.llm.with_structured_output(CounterfeitPrediction)

    def predict(self, product: pd.Series) -> bool:
        """
        Predict whether a single product is counterfeit.

        Args:
            product: A single product row as a pandas Series (without the label column)

        Returns:
            True if the product is likely counterfeit, False if genuine
        """
        if self.chain is None:
            print("Model not trained. Calling train()...")
            self._train()

        result = self.chain.invoke({"input": self._format_product(product)})
        return result.is_counterfeit
