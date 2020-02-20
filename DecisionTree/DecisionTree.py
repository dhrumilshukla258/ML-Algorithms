from HelperFunction import *

class DecisionTree:
    def __init__(self, attributes, labels, strMethod ):    
        self.parent = None
        self.nodeGainRation = float(0.0)
        self.nodeInformationGain = float(0.0)
        self.isLeaf = False
        self.majorityClass = int(0)
        self.bestAttribute = int(0)
        self.children = {}
        if strMethod == "Entropy":
            self.BuildTree( attributes, labels, strMethod, computeEntropy, InformationGainByEntropy )
        if strMethod == "VarianceImpurity":
            self.BuildTree( attributes, labels, strMethod, computeVarianceImpurity, InformationGainByVI )

    def BuildTree( self, attributes, labels, strMethod, ComputeMethod, InfoGainMethod ): 
        numInstances = len( labels )
        nodeInformation = numInstances * ComputeMethod(labels)
        self.majorityClass = mostFrequentlyOccurringValue( labels )
        
        if nodeInformation == 0:
            self.isLeaf = True
            return
        
        bestAttribute = None
        bestInformationGain = -math.inf
        loop = 0
        for X in attributes.T:
            infoGain = InfoGainMethod(labels,X)                  
            
            if infoGain > bestInformationGain:
                bestInformationGain = infoGain
                bestAttribute = loop
            loop += 1

        if bestInformationGain == -math.inf or bestInformationGain == 0:
            self.isLeaf = True
            return 
        
        # Otherwise split by the best attribute
        self.bestAttribute = bestAttribute
        self.nodeInformationGain = bestInformationGain
        UniqueValuesInX =  np.unique( attributes[:,bestAttribute] )
        for Y in UniqueValuesInX:
            ids = segregate(attributes[:,bestAttribute], Y)
            self.children[Y] =  DecisionTree(attributes[ids], labels[ids], strMethod )
            self.children[Y].parent = self
        return        

    def RemoveChild(self, twig):
        if self == twig:
            self.chidren = None  # Kill the chilren
            self.isLeaf = True
            self.nodeInformationGain = 0
            return  
        else:
            for val,child in self.children.items():
                child.RemoveChild( twig )
            return

    def Evaluate(self, testAttributes):
        #print(testAttributes)
        if (self.isLeaf):
            return self.majorityClass
        else:
            return self.children[
                testAttributes[self.bestAttribute]
            ].Evaluate(testAttributes)
        
    def PrintRec(self, columnName, ans, bestAttribute):
        if (self.isLeaf):
            ans.append( str(self.majorityClass) ) 
            return ans
        else:
            for val,child in self.children.items():
                ans.append ( str( columnName[bestAttribute] ) + " = " + str( val ) + " : " )
                self.children[val].PrintRec( columnName, ans, self.children[val].bestAttribute )
            
            return ans
            
    def PrintTree(self, columnName ):
        ans = []
        ans = self.PrintRec( columnName, ans, self.bestAttribute)
        answer = ""
        loop = 0
        dic_loop = {} 
        for i in ans:
            temp = True
            if ":" not in i:
                answer+=i
                temp = False
            
            if temp:
                if dic_loop.get(i[0:2]) == None:
                    dic_loop[ i[0:2] ] = loop
                else:
                    loop = dic_loop.get(i[0:2])

                answer += "\n"
                for j in range(loop):
                    answer+= "| "
                answer += i
            loop +=1
        print(answer)


def CalculateAccuracy( dt, X, Y):
    predict_Y = []
    for i in range( len(X) ):
        predict_Y.append( dt.Evaluate( X[i,:] ) )

    no_of_matches = 0
    for i in range( len(Y) ):
        if predict_Y[i] == Y[i]:
            no_of_matches+=1

    accuracy = no_of_matches/len(Y)
    
    #Accuracy Calculation
    return accuracy