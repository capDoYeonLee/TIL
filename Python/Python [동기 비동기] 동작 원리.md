# Python [동기 / 비동기] 동작 원리

동기 VS 비동기 (Python 3.4에서 asyncio가 표준 라이브러리로 추가)

```python
def sub_routine():
	print("It is subroutine")

async def co_routine():
	print("It is coroutine")
```

<img src=”./Untitled.png”>

e.g)

```python
if __name__ == "__main__":
	ret = getIp("www.google.com")
	print("Hello")
```

- 동기 방식 : getIp()의 결과값을 받을 때까지 getIp()에서 메인 쓰레드가 작동한다.
- 비동기 방식 : getIp()의 결과값을 전달받는 것과 상관없이, 바로 pinrt()를 실행한다.

<img src=”./Untitled1.png”>

<aside>
💡 제너레이터는 다음과 같이 선언 가능함. 
기본 함수와 동일하게 선언하지만, 내부적인 yield키워드를 가지면 제너레이터로 만들어진다.
일반적인 함수를 실행하는 것처럼 제너레이터를 실행하면, 
함수(서브루틴)이 실행되는 것이 아니라 제너레이터 객체를 반환받는다. 
반환받은 제너레이터 객체를 next() / send()메서드를 이용해서 실행할 수 있다.

</aside>