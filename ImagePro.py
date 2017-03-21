import numpy as np  
import cPickle  
import cv2  
import glob

# construct the argument parser and parse the arguments  
class RGBHistogram:  
    def __init__(self, bins):  
        # store the number of bins the histogram will use  
        self.bins = bins  
   
    def describe(self, image):  
        # compute a 3D histogram in the RGB colorspace,  
        # then normalize the histogram so that images  
        # with the same content, but either scaled larger  
        # or smaller will have (roughly) the same histogram  
        hist = cv2.calcHist([image], [0, 1, 2],  
            None, self.bins, [0, 256, 0, 256, 0, 256])  
        cv2.normalize(hist,hist)  
        # return out 3D histogram as a flattened array  
        return hist.flatten()  

class Searcher:  
    def __init__(self, index):  
        # store our index of images  
        self.index = index  
  
    def search(self, queryFeatures):  
        # initialize our dictionary of results  
        results = {}  
  
        # loop over the index  
        for (k, features) in self.index.items():  
            # compute the chi-squared distance between the features  
            # in our index and our query features -- using the  
            # chi-squared distance which is normally used in the  
            # computer vision field to compare histograms  
            d = self.chi2_distance(features, queryFeatures)  
  
            # now that we have the distance between the two feature  
            # vectors, we can udpate the results dictionary -- the  
            # key is the current image ID in the index and the  
            # value is the distance we just computed, representing  
            # how 'similar' the image in the index is to our query  
            results[k] = d  
  
        # sort our results, so that the smaller distances (i.e. the  
        # more relevant images are at the front of the list)  
        results = sorted([(v, k) for (k, v) in results.items()])  
  
        # return our results  
        return results  
  
    def chi2_distance(self, histA, histB, eps = 1e-10):  
        # compute the chi-squared distance  
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)  
            for (a, b) in zip(histA, histB)])  
  
        # return the chi-squared distance  
        return d     
    
    
if __name__ == '__main__':    
    args = {"dataset":"D:\ImageProcessing\imagenet\\ImageData",
            "index":"D:\ImageProcessing\imageSearch\Test.txt",
             "query":"D:\ImageProcessing\imagenet\ImageData\\09229709_34868.jpeg"
           }
    index = {}  
    
    desc = RGBHistogram([8, 8, 8])  
    for imagePath in glob.glob(args["dataset"] + "/*.JPEG"):  
        # extract our unique image ID (i.e. the filename)  
        k = imagePath[imagePath.rfind("/") + 1:]    
        # load the image, describe it using our RGB histogram  
        # descriptor, and update the index  
        image = cv2.imread(imagePath) 
        features = desc.describe(image) 
        index[k] = features  
    f = open(args["index"], "w")  
    f.write(cPickle.dumps(index))  
    f.close()   
    
    queryImage = cv2.imread(args["query"])  
    cv2.imshow("Query", queryImage)  
    print "query: %s" % (args["query"]) 
    
    desc = RGBHistogram([8, 8, 8])  
    queryFeatures = desc.describe(queryImage)    
    # load the index and initialize our searcher  
    index = cPickle.loads(open(args["index"]).read())
    searcher = Searcher(index)  
    # loop over images in the index -- we will use each one as  
    # a query image  
    results = searcher.search(queryFeatures)  
    
    montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")  
    montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8") 
    
    
    for j in xrange(0, 10):  
            # grab the result (we are using row-major order) and  
            # load the result image  
        (score, imageName) = results[j]
        print imageName 
        path = imageName
        result = cv2.imread(path)  
        print "\t%d. %s : %.3f" % (j + 1, imageName, score)  
      
            # check to see if the first montage should be used  
        if j < 5:  
             montageA[j * 166:(j + 1) * 166, :] = result  
      
            # otherwise, the second montage should be used  
        else:  
            montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result  
      
        # show the results  
    cv2.imshow("Results 1-5", montageA)  
    cv2.imshow("Results 6-10", montageB)  
    cv2.waitKey(0)  
