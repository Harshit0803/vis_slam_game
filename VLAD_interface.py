from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import statistics 

import itertools
import cv2

class VLAD(object):
    """
    This class represents the VLAD (Vector of Locally Aggregated Descriptors) model.
    """
    def __init__(self):
        # Sets ORB as the default descriptor
        self.descriptorName = "ORB" 

        # List to hold the results of image queries
        self.queryResults = []  

        # Structure for efficient indexing of descriptors
        self.indexStructure = []  

        # Holds the visual dictionary used for VLAD descriptors
        self.visualDictionary = None 
        
        # Threshold for the minimum variance
        self.min_variance = 1e6  

    def train(self, train_imgs):
        """
        Trains the VLAD model using a set of training images.
        """
        # Fetching descriptors based on the specified type
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
        descriptors = getDescriptors(train_imgs, dict[self.descriptorName])

        # Creating the visual dictionary using K-means
        k = 16  # Number of clusters for the dictionary
        self.visualDictionary = kMeansDictionary(descriptors, k)

        # Generating VLAD descriptors
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
        V, imageID = getVLADDescriptors(train_imgs, dict[self.descriptorName], self.visualDictionary)

        # Creating a Ball Tree for efficient querying
        leafSize = 40  # Parameter for Ball Tree construction
        tree = indexBallTree(V, leafSize)
        self.indexStructure = [imageID, tree]
        
        return(0)

    def query(self):
        """
        Performs a query operation on the VLAD model with multiple images.
        """
        def query_single_image(path, k, descriptorName):
            """
            Handles querying of a single image.
            """
            tree = self.indexStructure[1]  # Accessing the Ball Tree

            # Retrieving descriptors and nearest neighbors
            _, ind = query(path, k, descriptorName, self.visualDictionary, tree)

            # Processing the indices to find variance
            ind = list(itertools.chain.from_iterable(ind))
            variance = statistics.variance(ind)
            
            # Updating results based on variance
            if variance < self.min_variance: 
                self.min_variance = variance
                self.queryResults.append(statistics.median(ind))

        # Nearest neighbors count
        k = 5

        # Specifying the paths for query images
        paths = ['/home/harshit/vis_nav_player/finGame/src/queries/0_img.png','/home/harshit/vis_nav_player/finGame/src/queries/1_img.png','/home/harshit/vis_nav_player/finGame/src/queries/2_img.png','/home/harshit/vis_nav_player/finGame/src/queries/3_img.png']

        # Querying each image
        for path in paths:
            query_single_image(path, k, self.descriptorName)
        
        # Output the results
        print("queryResults: ", self.queryResults)
        
        # Return the query results in reverse order
        return self.queryResults[::-1]
