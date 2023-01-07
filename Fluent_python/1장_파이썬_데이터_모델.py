import collections
# collections 모듈은 파이썬 자료형(list, tuple, dict)들에게 확장된 기능을 주기 위해 제작된 파이썬 내장 모듈이다.
# 자주 쓰는 클래스는 3가지가 있고 알아두면 좋을만한 것 3가지가 있다.
# Counter, deque, defaultdict <자주 쓰는 클래스>
# OrderedDict, namedtuple, ChainMap

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self.__cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]


    def __len__(self):
        return len(self.__cards)


    def __getitem__(self, position):
        return self.__cards[position]

deck = FrenchDeck()
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
	print(card)