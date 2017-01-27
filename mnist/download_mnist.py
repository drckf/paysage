import urllib.request, os
filepath = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(filepath,"mnist.h5")):
    urllib.request.urlretrieve("https://sites.google.com/site/charleskennethfisher/home/programs-and-data/mnist.h5?attredirects=0&d=1",
                              os.path.join(filepath, "mnist.h5"))
