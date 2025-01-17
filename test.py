from unittest import TestCase
from unittest.mock import Mock
from main import SearchEngine

class TestSearchEngine(TestCase):
    def test_load_data(self):
        instance = SearchEngine("Vilken KVÅ-kod använder man när man deltagit i ett SIP möte?")
        expectation = list
        self.assertIsInstance(instance.load_data(), expectation)

    def test_split_data(self):
        instance = SearchEngine("Vilken KVÅ-kod använder man när man deltagit i ett SIP möte?")
        expectation = list
        self.assertIsInstance(instance.split_data(instance.load_data()), expectation)

    def test_add_documents_to_vector_store(self):
        instance = SearchEngine("Vilken KVÅ-kod använder man när man deltagit i ett SIP möte?")
        instance.vector_store = Mock()
        data = instance.load_data()
        all_splits = instance.split_data(data)
        instance.add_documents_to_vector_store(all_splits)
        instance.vector_store.add_documents.assert_called()

    def test_retrieve_documents(self):
        instance = SearchEngine("Vilken KVÅ-kod använder man när man deltagit i ett SIP möte?")
        instance.vector_store = Mock()
        data = instance.load_data()
        all_splits = instance.split_data(data)
        instance.add_documents_to_vector_store(all_splits)
        instance.vector_store.search(query=instance.question, search_type="mmr")
        instance.vector_store.search.assert_called()

    def test_get_answer(self):
        instance = SearchEngine("Vilken KVÅ-kod använder man när man deltagit i ett SIP möte?")
        data = instance.load_data()
        all_splits = instance.split_data(data)
        instance.add_documents_to_vector_store(all_splits)
        retrieved = instance.vector_store.search(query=instance.question, search_type="mmr")
        self.assertIsInstance(instance.get_answer(retrieved), str)

    def test_main(self):
        instance = SearchEngine("Vilken KVÅ-kod använder man när man deltagit i ett SIP möte?")
        self.assertIsInstance(instance.main(), str)






