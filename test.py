

class Parent:
    def __init__(self):
        self.x = 110
    def greet(self, name="Alice"):
        print(f"Hello, {name}!")

class Child(Parent):
    def __init__(self):
        super().__init__()  # Call the parent constructor
        self.x = 120
        print(self.x)  # This will print 120, the child's x value
    def greet(self, name="Bob"):  # Overrides default argument
        super().greet(name)


# Example usage
if __name__ == "__main__":
    child = Child()  # This will print 120
    child.greet()  # This will print "Hello, Bob!"
    child.greet("Charlie")  # This will print "Hello, Charlie!"