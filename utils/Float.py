class Float(float):
    add_counter = 0
    sub_counter = 0
    mul_counter = 0
    div_counter = 0

    def __new__(cls, value):
        return super().__new__(cls, value)
    
    def __add__(self, value):
        Float.add_counter += 1
        return Float(super().__add__(value))
    
    def __radd__(self, value):
        Float.add_counter += 1
        return Float(super().__radd__(value))
    
    def __sub__(self, value):
        Float.sub_counter += 1
        return Float(super().__sub__(value))
    
    def __rsub__(self, value):
        Float.sub_counter += 1
        return Float(super().__rsub__(value))
    
    def __mul__(self, value):
        Float.mul_counter += 1
        return Float(super().__mul__(value))
    
    def __rmul__(self, value):
        Float.mul_counter += 1
        return Float(super().__rmul__(value))
    
    def __truediv__(self, value):
        Float.div_counter += 1
        return Float(super().__truediv__(value))
    
    def __rtruediv__(self, value):
        Float.div_counter += 1
        return Float(super().__rtruediv__(value))
    
    @staticmethod
    def reset():
        Float.add_counter = 0
        Float.sub_counter = 0
        Float.mul_counter = 0
        Float.div_counter = 0