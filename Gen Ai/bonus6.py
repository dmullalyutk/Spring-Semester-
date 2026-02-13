# Run the following for use in Q1 and Q2:

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from transformers.pipelines import PIPELINE_REGISTRY

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    torch_dtype="auto")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

supported_tasks = PIPELINE_REGISTRY.get_supported_tasks()
task = "text2text-generation" if "text2text-generation" in supported_tasks else "text-generation"

if task == "text-generation":
    generator = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
else:
    generator = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
    )


############ Question 1:
#Find a way to extract the following information from the text below using an LLM.
#What is the name of the product?
#Who developed the product?
#What is the annual global sales?
#When was the product introduced?

snickers = """Snickers is a chocolate bar consisting of nougat topped with caramel and peanuts,
all encased in milk chocolate.The bars are made by the American company Mars Inc.
The annual global sales of Snickers is over $380 million, and it is widely
considered the bestselling candy bar in the world. Snickers was introduced by Mars in 1930
and named after the Mars family's favorite horse.
Initially marketed as "Marathon" in the UK and Ireland, its name was changed to Snickers
in 1990 to align with the global brand, differentiating it from an unrelated US product
also named Marathon. Snickers has expanded its product line to include variations such
as mini, dark chocolate, white chocolate, ice cream bars, and several nut, flavor, and
protein-enhanced versions. Ingredients have evolved from its original formulation to
adapt to changing consumer preferences and nutritional guidelines. Despite fluctuations
in bar size and controversies around health and advertising, Snickers remains a prominent
snack worldwide, sponsoring significant sporting events and introducing notable marketing
campaigns."""

#Answer:
q1_prompt = f"""
Extract the following fields from the text and return exactly this format:
Product Name: <value>
Developer: <value>
Annual Global Sales: <value>
Introduced: <value>

Text:
{snickers}
"""

q1_output = generator(q1_prompt)[0]["generated_text"]
print("Q1 Extracted Information:")
print(q1_output)


############  Question 2 - chain prompting
#Create a chain prompt.
# In the first prompt instruct the model to create a new name for cookies.
# In the second prompt instruct the model to generate a list of ingredients for cookies with that specific brand name.
# In the final prompt, use the brand name and ingredients and instruct the model to explain how the cookies are different from other cookies.
# Do not hard code name and ingredients, use variables instead.

#Answer:

# Step 1: Create a cookie brand name
step1_prompt = "Create one creative brand name for a new cookie product. Return only the brand name."
brand_name = generator(step1_prompt)[0]["generated_text"].strip()
print("\nGenerated Brand Name:", brand_name)

# Step 2: Generate ingredients using that brand name
step2_prompt = f"Generate a concise ingredient list for a cookie brand named '{brand_name}'. Return comma-separated ingredients only."
ingredients = generator(step2_prompt)[0]["generated_text"].strip()
print("Ingredients:", ingredients)

# Step 3: Explain how this cookie is different using both variables
step3_prompt = (
    f"Brand name: {brand_name}\n"
    f"Ingredients: {ingredients}\n"
    "Explain in 3-4 sentences how this cookie is different from other cookies."
)

differentiation = generator(step3_prompt)[0]["generated_text"].strip()
print("\nHow it is different:")
print(differentiation)
