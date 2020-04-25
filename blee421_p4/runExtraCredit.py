from DataInterface import getExtraCreditDataset
from DecisionTree import makeTree, setEntropy, infoGain
from Testing import getAverageClassificaionRate, printDemarcation

'''
ABOUT THE DATASET

Source:                 http://archive.ics.uci.edu/ml/datasets/Poker+Hand
Name:                   Poker Hand Data Set
Abstract:               Purpose is to predict poker hands
Description:            Each card is described using two attributes (suit and rank),
                        for a total of 10 predictive attributes.
                        There is one Class attribute that describes the "Poker Hand".
                        The order of cards is important,
                        which is why there are 480 possible Royal Flush hands as compared to 4.
Attributes:             Si "Suit of card #i"
                        Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
                        Ci "Rank of card #i"
                        Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
Class Label:            "Poker Hand"
                        Ordinal (0-9)
                        0: Nothing in hand; not a recognized poker hand
                        1: One pair; one pair of equal ranks within five cards
                        2: Two pairs; two pairs of equal ranks within five cards
                        3: Three of a kind; three equal ranks within five cards
                        4: Straight; five cards, sequentially ranked with no gaps
                        5: Flush; five cards with the same suit
                        6: Full house; pair + different rank three of a kind
                        7: Four of a kind; four equal ranks within five cards
                        8: Straight flush; straight + flush
                        9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
'''


def testPoker(setFunc=setEntropy, infoFunc=infoGain):
    examples, attrValues, labelName, labelValues = getExtraCreditDataset()
    print 'Testing Extra Credit dataset. Number of examples %d.' % len(examples)
    tree = makeTree(examples, attrValues, labelName, setFunc, infoFunc)
    f = open('poker.out', 'w')
    f.write(str(tree))
    f.close()
    print 'Tree size: %d.\n' % tree.count()
    print 'Entire tree written out to poker.out in local directory\n'
    evaluation = getAverageClassificaionRate((examples, attrValues, labelName, labelValues))
    print 'Results for training set:\n%s\n' % str(evaluation)
    printDemarcation()
    return tree, evaluation


testPoker()
