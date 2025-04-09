import json
import os
from typing import Union, Optional

class Prompter(object):
    def __init__(self, template_name: Optional[str] = None, verbose: bool = False) -> None:
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        
        file_name = os.path.join("tools/prompt_template", f"{template_name}.json")
        if os.path.exists(file_name):
            with open(file=file_name) as json_file:
                self.template = json.load(json_file)
        else:
            raise FileNotFoundError(f"Can't open {file_name}")
        
        if self._verbose:
            print(
                f'Using prompt template {template_name}: {self.template["description"]}'
            )
        
    def generate_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        label: Optional[str] = None
    ):
        if input:
            res = self.template["prompt_input"].format(
                instruction = instruction,
                input = input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction = instruction
            )
        
        if label:
            res = f"{res}{label}"
        
        if self._verbose:
            print(res)

        return res

    def get_response(self, output: str):
        '''
        **__Args:__**
        - output: output response from LLMs
        '''
        return output.split(self.template["response_split"])[1].strip()
