import numpy as np

from unittest import TestCase
from tree_isomorphism import Graph, Tree


class TestGraphs(TestCase):
    def setUp(self):
        self.graph_one_node = Graph(np.array([[0]]))
        self.graph_tree_even_diameter = Graph(
            np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        )
        self.graph_tree_odd_diameter = Graph(
            np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        )
        self.graph_not_tree = Graph(
            np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
                      [0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
                      [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        )


class TestTreeInit(TestGraphs):

    def test_graph_one_node__valid_tree(self):
        tree = Tree(self.graph_one_node)
        self.assertEqual(tree.root, 0)
        self.assertEqual(tree.adj_list, [[]])

    def test_graph_even_diam__valid_tree(self):
        tree = Tree(self.graph_tree_even_diameter)
        self.assertEqual(tree.root, -1)
        self.assertEqual(set(tree.adj_list[-1]), {0, 1})
        self.assertEqual(set(tree.adj_list[0]), {2, 3, -1})
        self.assertEqual(set(tree.adj_list[1]), {5, 6, 7, -1})

    def test_graph_odd_diam__valid_tree(self):
        tree = Tree(self.graph_tree_odd_diameter)
        self.assertEqual(tree.root, 9)
        self.assertEqual(set(tree.adj_list[9]), {0, 1})
        self.assertEqual(set(tree.adj_list[0]), {2, 3, 9})
        self.assertEqual(set(tree.adj_list[1]), {5, 6, 7, 9})

    def test_graph_not_tree__no_tree(self):
        tree = Tree(self.graph_not_tree)
        self.assertEqual(tree.root, None)


class TestTree(TestCase):
    def setUp(self):
        self.tree_one_node = Tree(Graph(np.array([[]])))
        self.tree_one_node.root = 0
        self.tree_one_node.adj_list = [[]]

        self.tree_even_diam = Tree(Graph(np.array([[]])))
        self.tree_even_diam.root = -1
        self.tree_even_diam.adj_list = [
            [-1, 2, 3],     # 0
            [-1, 5, 6, 7],  # 1
            [0, 4],         # 2
            [0],            # 3
            [2],            # 4
            [1],            # 5
            [1],            # 6
            [1, 8],         # 7
            [7],            # 8
            [0, 1]          # -1
        ]

        self.tree_odd_diam = Tree(Graph(np.array([[]])))
        self.tree_odd_diam.root = 9
        self.tree_odd_diam.adj_list = [
            [2, 3, 9],      # 0
            [5, 6, 7, 9],   # 1
            [0, 4],         # 2
            [0],            # 3
            [2],            # 4
            [1],            # 5
            [1],            # 6
            [1, 8],         # 7
            [7],            # 8
            [0, 1]          # 9
        ]

        self.not_tree = Tree(Graph(np.array([[]])))
        self.not_tree.root = None
        self.not_tree.adj_list = None


class TestTreeCompleteInvariant(TestTree):

    def test_tree_one_node__valid_invariant(self):
        self.assertEqual(self.tree_one_node.complete_invariant(), (1, [1, 0]))

    def test_not_tree__no_invariant(self):
        self.assertEqual(self.not_tree.complete_invariant(), None)

    def test_tree_even_diam__valid_invariant(self):
        self.assertEqual(
            self.tree_even_diam.complete_invariant(), (-1, [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0])
        )

    def test_tree_odd_diam__valid_invariant(self):
        self.assertEqual(
            self.tree_odd_diam.complete_invariant(), (1, [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0])
        )
