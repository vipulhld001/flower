import hydra
from hydra.utils import call, instantiate
from omegaconf import OmegaConf, DictConfig




def function_test(x,y):
    result = x + y
    print(f"{result = }")
    return result

class MyClass:
    def __init__(self,x) -> None:
        self.x = x

    def printxsquared(self):
        print(f"{self.x**2 = }")
        
class MyComplexClass:
    def __init__(self, my_object: MyClass):
        self.obj = my_object
    
    def instantiate_obj(self):
        self.obj = instantiate(self.obj)



@hydra.main(config_path="conf",config_name="toy2", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print(cfg.foo)
    print(cfg.bar.more)
    ### Calling a Function

    output = call(cfg.my_func)
    print(f"{output = }")


    output = call(cfg.my_func, y=33)
    print(f"{output = }")

    print("Partials")

    fn = call(cfg.my_partial_func)

    output = fn(y =199)

    #####
    ##class

    print("Objects (Classes)")

    obj = instantiate(cfg.my_object)

    obj.printxsquared()

#Complicated object (Object inside object)
    print("_____"*20)

    complex_obj= instantiate(cfg.my_complex_object)
    
    complex_obj.instantiate_obj()
    print(complex_obj.obj.printxsquared())
    #####Print Toy MOdel

    print(cfg.toy_model)
    mymodel = instantiate(cfg.toy_model)
    print(mymodel)





















if __name__ == "__main__":
    

    main()
    
