# Design Architecture of Prompter

Ultimately, LLMs are stateless functions of data.  They are nondeterministic, so not pure functions, but they are static and don't modify any state.

Because they are stateless functions, we can get a lot of mileage out of functional patterns.

Some thoughts:
- We can do arbitrary transformations on prompt data before it arrives at the LLM.
- We can do arbitrary transformations on the LLM response as well.
- Adding state to LLM interfaces, which some APIs are starting to do, is best avoided.
- Even today, tool calling is a result of these transformations of prompts on the way in and out of the LLM.  The tools are converted into text instructions about how to indicate that they should be used, and the output is parsed for any sign of those instructions, which are taken out of the text and plopped back into a tool call result.
- We could do all sorts of similar things ourselves.
- Pydantic is nifty, it's a great way to tie together schema definition with parsing.
- JSON schema is deep in the posttraining steps of LLMs, so we should keep leaning on it

This library attempts to provide a standard object structure for representing prompts and responses.

By writing transforms from this structure into the various LLM APIs, we can write prompts once and run them against any LLM we want.

But also, we could write OTHER schemas, which transform into this structure.  Which could let us write tiny DSLs for LLM interactions.

I would love to see experiments with different ways of doing things like tool calling, I think it is possible.