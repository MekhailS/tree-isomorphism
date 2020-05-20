import numpy as np

from unittest import TestCase
from tree_isomorphism import Graph


class TestGraphInit(TestCase):

    def test_empty_matrix__empty_adj_list(self):
        matrix = np.array([[]])
        graph = Graph(matrix)
        self.assertTrue(not graph.adj_list)

    def test_1d_matrix__empty_adj_list(self):
        matrix = np.array([1, 0])
        graph = Graph(matrix)
        self.assertTrue(not graph.adj_list)

    def test_3d_matrix__empty_adj_list(self):
        matrix = np.array([[[0, 1]], [[1, 0], [1, 0]]])
        graph = Graph(matrix)
        self.assertTrue(not graph.adj_list)

    def test_2x3_matrix__empty_adj_list(self):
        matrix = np.array([[0, 1, 0],
                           [1, 0, 0]])
        graph = Graph(matrix)
        self.assertTrue(not graph.adj_list)

    def test_not_sym_matrix__empty_adj_list(self):
        matrix = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 1]])
        graph = Graph(matrix)
        self.assertTrue(not graph.adj_list)

    def test_valid_1x1_matrix__valid_adj_list(self):
        matrix = np.array([[0]])
        graph = Graph(matrix)
        self.assertEqual(len(graph.adj_list), 1)

    def test_valid_3x3_matrix__valid_adj_list(self):
        matrix = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
        graph = Graph(matrix)
        self.assertEqual(set(graph.adj_list[0]), {1})
        self.assertEqual(set(graph.adj_list[1]), {0, 2})
        self.assertEqual(set(graph.adj_list[2]), {1})


class TestGraph(TestCase):
    def setUp(self):
        self.graph_tree = Graph(np.array([[]]))
        self.graph_unlinked = Graph(np.array([[]]))
        self.graph_loop = Graph(np.array([[]]))
        self.graph_one_node = Graph(np.array([[]]))

        # adjacency matrix of a tree
        self.graph_tree.adj_list = [
            [1, 2],     # 0
            [3, 4, 0],  # 1
            [0, 5],     # 2
            [1, 6],     # 3
            [1],        # 4
            [2],        # 5
            [3]         # 6
        ]
        # adjacency matrix of a graph with loop
        self.graph_unlinked.adj_list = [
            [1, 2],     # 0
            [0],        # 1
            [0, 5],     # 2
            [4, 6],     # 3
            [3],        # 4
            [2],        # 5
            [3]         # 6
        ]
        # adjacency matrix of a graph with loop
        self.graph_loop.adj_list = [
            [1, 2],     # 0
            [0, 3, 4],  # 1
            [0, 3, 5],  # 2
            [1, 4, 6],  # 3
            [1, 3],     # 4
            [2],        # 5
            [3],        # 6
        ]
        # adjacency matrix of a graph with one vertex
        self.graph_one_node.adj_list = [[0]]


class TestGraphDFS(TestGraph):

    def test_graph_tree_from0__tree_furthest6(self):
        is_tree, furthest = self.graph_tree.dfs(0)
        self.assertTrue(is_tree)
        self.assertEqual(furthest, 6)

    def test_graph_tree_from6__tree_furthest5(self):
        is_tree, furthest = self.graph_tree.dfs(6)
        self.assertTrue(is_tree)
        self.assertEqual(furthest, 5)

    def test_graph_one_node__not_tree_furthest0(self):
        is_tree, furthest = self.graph_one_node.dfs(0)
        self.assertFalse(is_tree)
        self.assertEqual(furthest, 0)

    def test_graph_unlinked_from6__not_tree_furthest4(self):
        is_tree, furthest = self.graph_unlinked.dfs(6)
        self.assertFalse(is_tree)
        self.assertEqual(furthest, 4)

    def test_graph_unlinked_from0__not_tree_furthest5(self):
        is_tree, furthest = self.graph_unlinked.dfs(0)
        self.assertFalse(is_tree)
        self.assertEqual(furthest, 5)

    def test_graph_loop_from0__not_tree(self):
        is_tree, _ = self.graph_loop.dfs(0)
        self.assertFalse(is_tree)

    def test_graph_loop_from3__not_tree(self):
        is_tree, _ = self.graph_loop.dfs(3)
        self.assertFalse(is_tree)


class TestGraphFindPath(TestGraph):

    def test_graph_tree_0to0__valid_path(self):
        path = self.graph_tree.find_path(0, 0)
        self.assertEqual(path, [0])

    def test_graph_tree_0to6___valid_path(self):
        path = self.graph_tree.find_path(0, 6)
        self.assertEqual(path, [0, 1, 3, 6])

    def test_graph_tree_2to3___valid_path(self):
        path = self.graph_tree.find_path(2, 3)
        self.assertEqual(path, [2, 0, 1, 3])

    def test_graph_tree_0to_invalid__no_path(self):
        invalid_node = 5112
        path = self.graph_tree.find_path(0, invalid_node)
        self.assertFalse(path)

    def test_graph_one_node_0to0__valid_path(self):
        path = self.graph_one_node.find_path(0, 0)
        self.assertEqual(path, [0])

    def test_graph_unlinked_0to6__no_path(self):
        path = self.graph_unlinked.find_path(0, 6)
        self.assertFalse(path)

    def test_graph_unlinked_0to5__valid_path(self):
        path = self.graph_unlinked.find_path(0, 5)
        self.assertEqual(path, [0, 2, 5])

    def test_graph_unlinked_3to4__valid_path(self):
        path = self.graph_unlinked.find_path(3, 4)
        self.assertEqual(path, [3, 4])

    def test_graph_loop_0to4__path(self):
        path = self.graph_loop.find_path(0, 6)
        self.assertTrue(path)

    def test_graph_loop_3to4__path(self):
        path = self.graph_loop.find_path(3, 4)
        self.assertTrue(path)
