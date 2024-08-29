def wrapper(func,newarg):
    print(newarg)
    def newfunc(*args,**kwargs): 
        print(args)
        print(kwargs) 
        return func(*args,**kwargs)
    return newfunc

@wrapper
def myfunc(a,b,c,d):
    print(a)
    print(b)
    print(c)
    print(d) 
    return 5,1 

v = myfunc()
print("\n\n\n",v)
