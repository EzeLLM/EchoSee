import logger.logger as logger
from utils import utils
import CONSTANTS as CONST
from typing import Dict, Any, Optional, Union
log = logger.Logger("llm_chain")

class BaseLLMChain:
    """Base class for LLM chains to inherit from."""
    def __init__(self):
        pass
    
    def run(self, prompt: str, **kwargs) -> str:
        """Run the chain on a prompt and input."""
        raise NotImplementedError("Subclasses must implement run method")

    def _format_prompt(self, prompt: str, **kwargs) -> str:
        """Format the prompt with the inputs."""
        return prompt

class COTChain(BaseLLMChain):
    """A customized LLM chain that reuses COT functionality."""
    def __init__(self, provider: str, model_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.provider = provider
        self.llm = utils.high_performance_llm
        self.model_kwargs = model_kwargs or {}
        self.hidden_cot = [CONST.PROVIDER_DEEPSEEK, CONST.PROVIDER_OPENAI]
        self.cot_end_tag = CONST.COT_END_TAG
        self.cot_think_tag = CONST.COT_THINK_TAG
        
    def parse_cot(self, output: str) -> str:
        """
        Parse the Chain of Thought (COT) from the output.
        If the provider is in the hidden_cot list, return the output as is.
        Otherwise, extract the text after the last COT end tag.
        
        Args:
            output (str): The output string from the LLM
            
        Returns:
            str: The parsed output with COT removed
        """
        # If the provider hides COT tokens, return the output as is
        if self.provider in self.hidden_cot:
            return output
        
        # Find the last occurrence of the COT end tag
        last_end_tag_index = output.rfind(self.cot_end_tag)
        
        # If the end tag is found, return everything after it
        if last_end_tag_index != -1:
            # Add the length of the end tag to get the position right after it
            start_index = last_end_tag_index + len(self.cot_end_tag)
            return output[start_index:].strip()
        
        # If no end tag is found, return the original output
        return output
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        return self.llm.invoke(prompt, **self.model_kwargs)
        
    def predict(self, prompt: str, parse: bool = True, **kwargs) -> str:
        """Legacy predict method for compatibility with LangChain."""
        return self.run(prompt, parse=parse, **kwargs)
        
    def run(self, prompt: str, parse: bool = True, **kwargs) -> str:
        """
        Run the chain with the given prompt.
        
        Args:
            prompt (str): The prompt to run
            parse (bool): Whether to parse the COT from the output
            **kwargs: Additional keyword arguments
            
        Returns:
            str: The response from the LLM
        """
        formatted_prompt = self._format_prompt(prompt, **kwargs)
        response = self._call_llm(formatted_prompt)
        
        if parse:
            return self.parse_cot(response)
        return response
    
    def invoke(self, input_data: Union[str, Dict[str, Any]], parse: bool = True, **kwargs) -> str:
        """
        Modern invoke method compatible with newer LangChain versions.
        
        Args:
            input_data: Either a string prompt or a dictionary of inputs
            parse (bool): Whether to parse the COT from the output
            **kwargs: Additional keyword arguments
            
        Returns:
            str: The response from the LLM
        """
        if isinstance(input_data, dict):
            prompt = input_data.get("prompt", "")
        else:
            prompt = input_data
            
        return self.run(prompt, parse=parse, **kwargs)

# Example use with additional features like templating
class TemplatedCOTChain(COTChain):
    """LLM chain with templating capabilities."""
    def __init__(self, provider: str, template: str = "", model_kwargs=None):
        super().__init__(provider, model_kwargs)
        self.template = template
        
    def _format_prompt(self, prompt: str, **kwargs) -> str:
        """Format the prompt using the template."""
        if not self.template:
            return prompt
            
        formatted_template = self.template.replace("{prompt}", prompt)
        # Replace any other template variables with values from kwargs
        for key, value in kwargs.items():
            formatted_template = formatted_template.replace(f"{{{key}}}", str(value))
            
        return formatted_template


if __name__ == "__main__":
    # Example usage
    chain = COTChain(CONST.PROVIDER_OPENAI)
    print(chain.run("What is the capital of France?", parse=True))
    
    # Example with template
    template_chain = TemplatedCOTChain(
        CONST.PROVIDER_OPENAI,
        template="Where is {prompt}"
    )
    print(template_chain.run("What is the capital of Italy?", parse=True))