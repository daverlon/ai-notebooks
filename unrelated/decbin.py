

def decbin(n: int, r: int = 0) -> str:
  ret = ''
  while n != 0:
    r = n%2
    n = int(n/2)
    ret += str(r)
    decbin(n, r)
  return ret[::-1]
  
    
x = int(input('number: '))
print(decbin(x))
