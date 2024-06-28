import hydra
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf


#use hydra as proviging values from the cmd line

def function_test(x,y):
    result = x + y
    print(f"{result = }")
    return result 




@hydra.main(config_path="conf", config_name= "toy", version_base= None)
def main(cfg: DictConfig):

    #print config as yaml
    print(OmegaConf.to_yaml(cfg))

    #easy part
    print(cfg.foo)
    print(cfg.bar.baz)
    print(cfg.bar.more.blabla)

    #less easy

    output = call(cfg.my_func)
    print(f"{output = }")

    print("-----New OP______")
#TO replace the value of x or y... y in this case with 100 instead of 567
    output = call(cfg.my_func, y=100)
    print(f"{output = }")
    print("Partials")
    fn = call(cfg.my_partial_func)
    print(f"{fn = }")

    #missing argyument is y

    output = fn(y=1000)







if __name__ == "__main__":

    main()