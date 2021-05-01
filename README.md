# 2D External Visual Hulls in Python

## Running this code

To use this code, you will first need a python environment with the following dependencies available:

 * matplotlib
 * numpy

Running the file graphingVisHullTwoD.py will output the results of partitioning a scene of multiple polygons into separate visual regions. The original polygons will have a light-grey outline. Visual numbers are colour coded so that black is 0, red is 1, green is 2, blue is 3, etc. The black regions therefore form the visual hull.

To test your own shape, simply create your own Scene() object in the same way that the existing world1, world2, etc. are created in the middle of the graphinvVisHullTwoD.py file right now. I plan to, in the future, make this more interactive or to take in a .txt file as input, so that this is slightly more convenient.

When testing this out, I recommend you add multiple polygons to your world/scene, as the visual hull of a single 2D polygon is just its convex hull, which isn't particularly interesting compared to what happens with multiple polygons!

## Background

Implementing an algorithm for creation of 2D external visual hulls in Python, as preparation for creating 3D ones in a Blender plugin.

I hope to provide a better overview of this in a future README, but for now, I'm just going to quickly copy-paste and edit something I'd written about them in a different context:

The concept of a "[Visual Hull][4]" is what I was looking for regarding a mesh that doesn't have indents that do not impact its silhouette and just add complexity to the model. While the Wikipedia page, and a bunch of other resources, usually just refer to visual hulls created from a finite set of viewpoints, e.g. a set of real-life cameras, Laurentini, who introduced the term "Visual Hull", also introduces a term "External Visual Hull" to refer to one created using *every* possible viewpoint outside of the convex hull [1]. This will have the exact same silhouette as the original mesh when viewed from any position outside the convex hull at any angle, so it was exactly what I was looking for.

[1] also contains an algorithm for computing the external visual hull for polyhedra, but it is sorta brute-force and is O(n^12). In [2], Petitjean introduces a much more efficient algorithm for polyhedra, and Bottino and Laurentini also describe a more efficient algorithm in [3]. In addition to these algorithms for polyhedra, other works by Laurentini also discuss algorithms for doing the same for curved objects, surfaces of revolution, etc. I have yet to come across anyone actually *implementing* these algorithms, though.

If someone reading this wants to look these up for themselves, then in addition to the below citations, I would like to note that even if you don't normally have access to papers, e.g. through a post-secondary institution, all of these seem to be publicly available. You can download [1] from ResearchGate [here][5], download [2] from ResearchGate [here][6] (though I have no clue why different authors are listed for it than Petitjean), and [3] can be found in Google Books. 

## TODO for this README:
 * Add in images of what a visual hull "removes" versus what it keeps compared to a convex hull.

## Citations:

[1] Laurentini, Aldo. (1994). The Visual Hull Concept for Silhouette-Based Image Understanding. Pattern Analysis and Machine Intelligence, IEEE Transactions on. 16. 150-162. 10.1109/34.273735. 

[2] Petitjean, S. (1998). A computational geometric approach to visual hulls. International Journal of Computational Geometry & Applications, 8(04), 407-436.

[3] Bottino, Andrea & Laurentini, Aldo. (2006). Retrieval of Shape from Silhouette. 10.1016/S1076-5670(05)39001-X.

  [4]: https://en.wikipedia.org/wiki/Visual_hull
  [5]: https://www.researchgate.net/publication/3192242_The_Visual_Hull_Concept_for_Silhouette-Based_Image_Understanding
  [6]: https://www.researchgate.net/publication/2781058_A_Computational_Geometric_Approach_To_Visual_Hulls
  
 