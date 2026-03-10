import comfy.samplers

class SamplerSchedulerSettings:
    
    @classmethod
    def INPUT_TYPES(cls):
        schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
        return {
            "required": {
                "sampler_name": (samplers, {"default": "euler_ancestral"}),
                "scheduler": (schedulers, {"default": "sgm_uniform"}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("sampler_name", "scheduler",)
    FUNCTION = "get_settings"
    CATEGORY = "Forbidden Vision"

    def get_settings(self, sampler_name, scheduler):
        return (sampler_name, scheduler,)