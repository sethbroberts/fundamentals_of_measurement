
#import some libraries
import random
import itertools
import math
import copy

#partitionsets is a 3rd party library, needs to be installed
#see https://pypi.python.org/pypi/PartitionSets/0.1.1
#pip install PartitionSets
from partitionsets import partition
from partitionsets import ordered_set

#networkx is a 3rd party library, needs to be installed
#pip install networkx
import networkx as nx 



def calculate_number_of_partitions(n):
    "Given a number of elements in a set, calculate the number of partitions of that set (the Bell Number, see http://en.wikipedia.org/wiki/Dobinski%27s_formula"
    s = 0
    #we iterate through 100 instead of infinity, then we use the round function
    for k in range(100):
        s = s + (k**n / math.factorial(k))
    return round((1/math.exp(1)) * s)



def make_cartesian_product(S1, S2):
    "Given the sets S1 and S2, make and return the cartesian product S1 x S2"
    cartesian_product = []
    for item_x in S1:
        for item_y in S2:
            cartesian_product.append((item_x, item_y))
    cartesian_product.sort()        
    return cartesian_product



def make_random_relation(cartesian_product):
    "Given a cartesian product, return a random relation on S"
    number_of_pairs_in_relation = random.randint(1, len(cartesian_product))
    relation = random.sample(cartesian_product, number_of_pairs_in_relation)
    return relation



def s1_related_to_s2(s1, s2, relation):
    "Given 2 elements s1 and s2, and a relation R, return True if s1Rs2, False if not"
    if (s1, s2) in relation:
        return True
    else:
        return False



def make_equivalence_relations2(S):
    "Given set S, make all the equivalence relations of S"
    #first make the partitions
    partitions = make_partitions(S)
    equivalence_relations = []
    for p in partitions:
        equivalence_relation = []
        for block in p:
            for item_x in block:
                for item_y in block:
                    equivalence_relation.append((item_x, item_y))
        equivalence_relation.sort()            
        equivalence_relations.append(equivalence_relation)
    equivalence_relations.sort()
    return equivalence_relations



def randomly_pick_one(some_list):
    "Return a random choice from the list of all choices"
    ix = random.randint(0, len(some_list)-1)
    return some_list[ix]



def check_reflexive(S, relation):
    "Given a relation R, check reflexive property"
    expected_reflexive_elements = {}
    for element in S:
        expected_reflexive_elements[(element, element)] = 1
    for element in expected_reflexive_elements:
        if (element in relation) == False:
            return False
    return True

def check_symmetric(relation):
    "Given a relation R, check symmetric property"
    for (s1, s2) in relation:
        if ((s2, s1) in relation) == False:
            return False
    return True

def check_transitive(relation):
    "Given a relation R, check transitive property"
    for (s1, s2) in relation:
        for (s3, s4) in relation:
            if s2 == s3:
                if s1 != s4:
                    if ((s1, s4) in relation) == False:
                        return False
    return True



def check_mapping(relation):
    "Given a relation R, check whether it is a mapping"
    for s1, s2 in relation:
        for s3, s4 in relation:
            if s1 == s3 and s2 != s4:
                return False
    return True



def make_partitions(S):
    "Given a set S, make all the partitions of the set"
    partitions = partition.Partition(ordered_set.OrderedSet(S))
    return partitions 



def find_equivalence_class(element, relation):
    "Given an element and an equivalence relation, find equivalence class"
    equivalence_class = {}
    x1, y1 = relation[0]
    if type(x1) == type([1, 2]):
        for e1, e2 in relation:
            k1, k2 = str(e1), str(e2)
            if element == e1 or element == e2:
                equivalence_class[k1] = e1
                equivalence_class[k2] = e2
        final_list_redundant = []        
        for i in equivalence_class.keys():
            final_list_redundant.append(equivalence_class[i])
        final_list = remove_redundant_items(final_list_redundant)
        return(final_list)

    else:    
        for e1, e2 in relation:
            if element == e1 or element == e2:
                equivalence_class[e1] = 1
                equivalence_class[e2] = 1
        return sorted(list(equivalence_class.keys()))


def get_underlying_elements_from_relation(relation):
    "Given a relation, find the underlying elements of the set"
    underlying_elements_duplicates = []
    for e1, e2 in relation:
        underlying_elements_duplicates.append(e1)
        underlying_elements_duplicates.append(e2)
    underlying_elements = remove_redundant_items(underlying_elements_duplicates)
    underlying_elements.sort()
    return underlying_elements



def find_partition_from_equivalence_relation(equivalence_relation):
    "Given an equivalence relation, find the corresponding partition"
    S = get_underlying_elements_from_relation(equivalence_relation)
    blocks_with_duplicates = []
    for element in S:
        equivalence_class = find_equivalence_class(element, equivalence_relation)
        blocks_with_duplicates.append(equivalence_class)
    blocks = []
    for block in blocks_with_duplicates:
        if block not in blocks:
            blocks.append(block)
    return sorted(blocks)



def find_equivalence_relation_from_partition(partition):
    "Given a partition, find the corresponding equivalence relation"
    equivalence_relation_with_duplicates = []
    for block in partition:
        for e1 in block:
            for e2 in block:
                equivalence_relation_with_duplicates.append((e1, e2))
    equivalence_relation = []
    for e1, e2 in equivalence_relation_with_duplicates:
        if (e1, e2) not in equivalence_relation:
            equivalence_relation.append((e1, e2))
    return sorted(equivalence_relation)



def make_automorphisms(S):
    "Make automorphisms (permutations) of S, each represented as a mapping"
    automorphisms = []
    for p in itertools.permutations(S):
        automorphism = []
        for i, e1 in enumerate(S):
            automorphism.append((e1, p[i]))
        automorphisms.append(automorphism)
    return automorphisms



def compose_mappings(f, g):
    "Compose mappings f, g; apply g first, then f"
    composite_mapping = []
    for e1, e2 in g:
        for e3, e4 in f:
            if e2 == e3:
                composite_mapping.append((e1, e4))
    composite_mapping.sort()
    return composite_mapping



def make_cyclic_subgroup(automorphism):
    "Find the subset of automorphisms that make a subgroup with the given automorphism"
    subgroup_automorphisms = [automorphism]
    product = compose_mappings(automorphism, automorphism)
    while automorphism != product:
        subgroup_automorphisms.append(product)
        product = compose_mappings(automorphism, product)
    #NOTE: commented this out on 2/1/15, may cause bugs??    
    #subgroup_automorphisms.sort()    
    return subgroup_automorphisms



def remove_redundant_items(redundant_list):
    "Remove redundant items from a list"
    non_redundant_list = []
    for item in redundant_list:
        if item not in non_redundant_list:
            non_redundant_list.append(item)
    non_redundant_list.sort()
    return non_redundant_list


def make_all_cyclic_subgroups(automorphisms):
    "Given a set of automorphisms, make all the cyclic subgroups"
    cyclic_subgroups_redundant = []
    for automorphism in automorphisms:
        cyclic_subgroup = make_cyclic_subgroup(automorphism)
        cyclic_subgroups_redundant.append(cyclic_subgroup)
    cyclic_subgroups = remove_redundant_items(cyclic_subgroups_redundant)
    return cyclic_subgroups


def find_equivalence_relation_from_cyclic_subgroup(cyclic_subgroup):
    "Given a cyclic subgroup, find the corresponding equivalence relation"
    equivalence_relation_redundant = []
    for automorphism in cyclic_subgroup:
        for element in automorphism:
            equivalence_relation_redundant.append(element)
    equivalence_relation = remove_redundant_items(equivalence_relation_redundant)
    equivalence_relation.sort()
    return equivalence_relation


def find_partition_from_cyclic_subgroup(cyclic_subgroup):
    "Given a cyclic subgroup, find the corresponding partition"
    equivalence_relation = find_equivalence_relation_from_cyclic_subgroup(cyclic_subgroup)
    partition = find_partition_from_equivalence_relation(equivalence_relation)
    return partition 



def make_mapping_corresponding_to_equivalence_relation(equivalence_relation):
    "Given an equivalence relation, make a mapping such that R = R_f"
    partition = find_partition_from_equivalence_relation(equivalence_relation)
    mapping = []
    for i, block in enumerate(partition):
        for s in block:
            mapping.append((s, i))
    mapping.sort()
    return mapping 


def find_cyclic_subgroup_from_equivalence_relation(equivalence_relation):
    "Given an equivalence relation, find the corresponding subgroup G of the group of automorphisms"    
    seed_automorphism = []
    partition = find_partition_from_equivalence_relation(equivalence_relation)
    for block in partition:
        try:
            for i in range(len(block)):
                seed_automorphism.append((block[i], block[i+1]))
        except:
            seed_automorphism.append((block[-1], block[0]))
    seed_automorphism.sort()
    subgroup = make_cyclic_subgroup(seed_automorphism)
    return subgroup 



def make_natural_map(equivalence_relation):
    "Given an equivalence relation, make the natural map"
    partition = find_partition_from_equivalence_relation(equivalence_relation)
    natural_map = []
    for block in partition:
        for s in block:
            natural_map.append((s, block))
    natural_map.sort()
    return natural_map 



def make_correspondance_from_mapping_range_to_equivalence_classes(mapping):
    "Given a mapping, show correspondence between mapping range and equivalence classes of the associated equivalence relation"
    range2items = {}
    for s, r in mapping:
        if not r in range2items:
            range2items[r] = []
        range2items[r].append(s)
    correspondence = []
    for r in range2items:
        ec = range2items[r]
        correspondence.append((r, ec))
    correspondence.sort()
    return correspondence


def make_phi_bar(partition):
    "Given a quotient set (set of equivalence classes under an equivalence relation), make a map phi: S/R -> Y"
    phi = []
    for i, block in enumerate(partition):
        phi.append((block, i))
    phi.sort()
    return phi 


def er1_refines_er2(er1, er2):
    "Given 2 equivalence relations er1 and er2, return True if er1 refines er2, False otherwise"
    for s1, s2 in er1:
        if (s1, s2) not in er2:
            return False
    return True


def check_reflexivity(relation):
    "Check reflexivity, i.e, whether a relation refines itself"
    return er1_refines_er2(relation, relation)


def check_antisymmetry(relation1, relation2):
    "Check antisymmetry: if R1 refines R2 and R2 refines R1, R1 == R2" 
    if er1_refines_er2(relation1, relation2):
        if er1_refines_er2(relation2, relation1):
            return (relation1 == relation2)
    return True
    

def check_transitivity(r1, r2, r3):
    "Check transitivity: if R1 refines R2 and R2 refines R3, then R1 refines R3"  
    if er1_refines_er2(r1, r2):
        if er1_refines_er2(r2, r3):
            return er1_refines_er2(r1, r3)
    return True


def make_hasse_network(equivalence_relations):
    "Make a .dot file with Hasse Diagram for the equivalence relations"
    G = nx.DiGraph()
    for er1 in equivalence_relations:
        for er2 in equivalence_relations:
            if er1_refines_er2(er1, er2) and er1 != er2:
                p1 = str(find_partition_from_equivalence_relation(er1))
                p2 = str(find_partition_from_equivalence_relation(er2))
                directly_adjacent = True
                for er3 in equivalence_relations:
                    if er1 != er3 and er2 != er3:
                        if er1_refines_er2(er1, er3) and er1_refines_er2(er3, er2):
                            directly_adjacent = False
                if directly_adjacent:
                    G.add_edge(p1, p2)     
    return G 



def calculate_er_intersection(er1, er2):
    "Given 2 equivalence relations, calculate the equivalence relation that is their intersection"
    S = get_underlying_elements_from_relation(er1)
    er3_redundant = []
    for s1 in S:
        for s2 in S:
            if (s1, s2) in er1 and (s1, s2) in er2:
                er3_redundant.append((s1, s2))
    er3 = remove_redundant_items(er3_redundant)
    er3.sort()
    return er3 



def calculate_er_union(er1, er2):
    "Given 2 equivalence relations, calculate the equivalence relation that is their union"    
    er_union_redundant = []
    G = nx.DiGraph()
    for s1, s2 in er1:
        G.add_edge(s1, s2)
    for s1, s2 in er2:
        G.add_edge(s1, s2)
    for n1 in G.nodes():
        for n2 in G.nodes():
            if nx.has_path(G, n1, n2):
                er_union_redundant.append((n1, n2))
    er_union = remove_redundant_items(er_union_redundant)
    er_union.sort()
    return er_union



def find_partition_from_mapping(mapping):
    "Given a mapping, find the corresponding equivalence relation"
    cache = {}
    for s, x in mapping:
        if not str(x) in cache:
            cache[str(x)] = []
        cache[str(x)].append(s)
    partition = []
    for x in cache:
        block = cache[x]
        block.sort()
        partition.append(block)
    partition.sort()
    return partition 




def check_group(G):
    "Given G, check whether G is a group (satisfies closure, associativity, identity, inverse"

    #TODO: fix this
    possible_identity = G[0]

    closure = True
    for a1 in G:
        for a2 in G:
            if compose_mappings(a1, a2) not in G:
                closure = False 
            if compose_mappings(a1, a2) == a1:
                possible_identity = a2

    associative = True
    for a1 in G:
        for a2 in G:
            for a3 in G:
                if compose_mappings(a1, compose_mappings(a2, a3)) != compose_mappings(compose_mappings(a1, a2), a3):
                    associative = False  

    identity_exists = True
    for a in G:
        if compose_mappings(a, possible_identity) != a:
            identity_exists = False  
    if identity_exists:
        identity = possible_identity
    #TODO: fix this
    else:
        identity = G[0]

    inverse_exists = True
    for a1 in G:
        found_identity = False
        for a2 in G:
            if compose_mappings(a1, a2) == identity and compose_mappings(a2, a1) == identity:
                found_identity = True
                break
        if found_identity == False:
            inverse_exists = False
            break

    if closure and associative and identity_exists and inverse_exists:
        return True
    else:
        print('closure: {}'.format(closure))
        print('associative: {}'.format(associative))
        print('identity exists: {}'.format(identity_exists))
        print('inverse exists: {}'.format(inverse_exists))
        return False        


def make_equivalence_relations(S):
    "Given S, make all the equivalence relations on S"
    automorphisms = make_automorphisms(S) #uses itertools.permutations(S)
    subgroups = make_all_cyclic_subgroups(automorphisms) #uses make_cyclic_subgroup, see above
    equivalence_relations_redundant = [] #in some cases, multiple subgroups correspond to the same equivalence relation
    for subgroup in subgroups:
        er = find_equivalence_relation_from_cyclic_subgroup(subgroup) #get the equivalence relation corresponding to this subgroup
        equivalence_relations_redundant.append(er)
    equivalence_relations = remove_redundant_items(equivalence_relations_redundant) #remove redundant cases
    equivalence_relations.sort()
    return equivalence_relations


def make_mapping(S, X):
    "Given two sets, make a mapping between them"
    f = []
    for s in S:
        x = random.choice(X)
        f.append((s, x))
    return f     


def find_greatest_element(equivalence_relations):
    "Given a set of equivalence relations, find the greatest element (consisting of all of SxS)"    
    for er in equivalence_relations:
        p = find_partition_from_equivalence_relation(er)
        if len(p) == 1:
            return er


def find_least_element(equivalence_relations):
    "Given a set of equivalence relations, find the least element (consisting of pairs (s, s), i.e., equality"
    for er in equivalence_relations:
        p = find_partition_from_equivalence_relation(er)
        if len(p) == len(get_underlying_elements_from_relation(er)):
            return er            


def calculate_union_of_set_of_er(equivalence_relations):
    "Given a set of equivalence relations, calculate the union of all of them"
    union = equivalence_relations[0]
    for er in equivalence_relations[1:]:
        union = calculate_er_union(er, union)
    return union


def calculate_intersection_of_set_of_er(equivalence_relations):
    "Given a set of equivalence relations, calculate the intersection of all of them"
    intersection = equivalence_relations[0]
    for er in equivalence_relations[1:]:
        intersection = calculate_er_intersection(er, intersection)
    return intersection 


def find_collection_of_er_with_specified_intersection(intersection, equivalence_relations):
    "Given a specified intersection and the set of all equivalence relations, find a subset of equivalence relations whose intersection is equal to that specified"    
    number_of_equivalence_relations = len(equivalence_relations)
    number_to_choose = random.randint(1, number_of_equivalence_relations)
    collection = random.sample(equivalence_relations, number_to_choose)
    this_intersection = calculate_intersection_of_set_of_er(collection)
    while this_intersection != intersection:
        number_to_choose = random.randint(1, number_of_equivalence_relations)
        collection = random.sample(equivalence_relations, number_to_choose)
        this_intersection = calculate_intersection_of_set_of_er(collection)
    return collection 


def make_natural_projections(cartesian_product):
    "Given a cartiesian product, make the natural projections"
    pi_1, pi_2 = [], []
    for s1, s2 in cartesian_product:
        pi_1.append(((s1, s2), s1))
        pi_2.append(((s1, s2), s2))
    return pi_1, pi_2


def evaluate_mapping(mapping, x):
    "Given a mapping and an element in its domain, return the corresponding value from the range"
    for x1, y1 in mapping:
        if x == x1:
            return y1


def make_random_equivalence_relation(S):
    "Given a set S, make an equivalence relation randomly"
    #will do this by generating a random partition of the set
    indexes = range(0, random.randint(1, len(S)))
    mapping = []
    for s in S:
        ix = random.choice(indexes)
        mapping.append((s, ix))
    p = find_partition_from_mapping(mapping)
    er = find_equivalence_relation_from_partition(p)
    return er 



def calculate_intersection(s1, s2):
    "Given 2 set-like objects, calculate their intersection"
    intersection = []
    for e1 in s1:
        if e1 in s2:
            intersection.append(e1)
    intersection.sort()
    return intersection 


def make_inverse_relation(relation):
    "Given a relation, return its inverse. Neither the relation nor the inverse need be a mapping."
    inverse_relation = []
    for x, y in relation:
        inverse_relation.append((y, x))
    #depending on context, this might cause a bug:
    inverse_relation.sort()
    return inverse_relation



def check_if_S_is_cartesian_product(S, ER1, ER2):
    "Given a set S, and two equivalence relations, check whether we can express S as a cartesian product"
    p_1 = find_partition_from_equivalence_relation(ER1)
    p_2 = find_partition_from_equivalence_relation(ER2)

    each_R1_class_intersects_every_R2_class = True
    for block1 in p_1:
        for block2 in p_2:
            intersection = calculate_intersection(block1, block2)
            if len(intersection) == 0:
                each_R1_class_intersects_every_R2_class = False
                
    each_R2_class_intersects_every_R1_class = True
    for block2 in p_2:
        for block1 in p_1:
            intersection = calculate_intersection(block1, block2)
            if len(intersection) == 0:
                each_R2_class_intersects_every_R1_class = False
                
    intersection_er = calculate_er_intersection(ER1, ER2)
    intersection_p = find_partition_from_equivalence_relation(intersection_er)
    R1_intersection_R2_is_equality_relation = (len(intersection_p) == len(S))

    union_er = calculate_er_union(ER1, ER2)
    union_p = find_partition_from_equivalence_relation(union_er)
    R1_union_R2_is_trivial_relation = (len(union_p) == 1)  

    if (each_R1_class_intersects_every_R2_class and each_R1_class_intersects_every_R2_class) \
    and R1_intersection_R2_is_equality_relation \
    and R1_union_R2_is_trivial_relation:
        return True
    else:
        return False 


def make_random_equivalence_relation2(S):
    "Given a set S, make an equivalence relation randomly"
    #will do this by generating a random mapping of the set, then returning corresponding equivalence relation
    #first make a set, called indexes, for the mapping to map to
    #indexes is a Python range that looks like this: [0, n)
    #n is a random integer, chosen from [1, len(S)]
    indexes = list(range(0, random.randint(1, len(S))))
    #now ready to start building mapping; initialize it to empty list
    mapping = []
    #we need to ensure that all elements of indexes are mapped to at least once
    #otherwise certain mappings become artificially exceedingly rare, like the mapping that corresponds to equality
    #so, the first element of S maps to the first element of indexes,
    #the second element of S maps to the second element of indexes, and so on through the last element of indexes
    for i in indexes:
        mapping.append((S[i], i))
    #remember that len(indexes) <= len(S)
    #so for any element of S not yet mapped, just choose an image from indexes randomly
    for s in S[len(indexes):]:
        ix = random.choice(indexes)
        mapping.append((s, ix))
    p = find_partition_from_mapping(mapping)
    er = find_equivalence_relation_from_partition(p)
    return er



def make_factor_pairs(n):
    #Given an integer, make all factor pairs (e.g., 6 -> [(1, 6), (2, 3)])
    factor_pairs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factor_pairs.append((i, n//i))
    return factor_pairs    



def express_as_cartesian_product(S):
    "Given set S, express it as a cartesian product by finding equivalence relations on S with necessary properties"
    #make a copy of S, where we can gradually remove elements
    R = copy.deepcopy(S)
    #make_factor_pairs(6) returns [(1,6), (2,3)], etc.
    #a random choice from this list gives the number of x-classes and the number of members in each of these classes
    num_x_classes, num_x_members = random.choice(make_factor_pairs(len(S)))
    #make first partition / equivalence relation
    p1 = []
    for i in range(num_x_classes):
        block = random.sample(R, num_x_members)
        for s in block:
            R.remove(s)
        block.sort()    
        p1.append(block)
    #make a deep copy of p1, we will be deleting items from the blocks
    p1_copy = copy.deepcopy(p1)
    #make second partition / equivalence relation
    p2 = []
    for i in range(num_x_members):
        p2_block = []
        for block in p1_copy:
            s2 = random.choice(block)
            p2_block.append(s2)
            block.remove(s2)
        p2_block.sort()
        p2.append(p2_block)

    er1 = find_equivalence_relation_from_partition(p1)
    er2 = find_equivalence_relation_from_partition(p2)
    
    return er1, er2    



def make_random_automorphism(S):
    "Given a set S, make an automorphism at random"
    S_ = S.copy()
    random.shuffle(S_)
    A = []
    for i, s in enumerate(S):
        A.append((S[i], S_[i]))
    A.sort()
    return A



def make_random_subgroup_of_automorphisms(S):
    "Given a set S, make a random subgroup of the automorphisms of S"
    A = make_random_automorphism(S)
    G = make_cyclic_subgroup(A)
    return G 



def get_quotient_set(R):
    "Given an equivalence relation, return the quotient set of S under R, S/R"
    return find_partition_from_equivalence_relation(R)



def make_random_mapping(S):
    "Given a set S, make a mapping randomly"
    #first make a set, called indexes, for the mapping to map to
    #indexes is a Python range that looks like this: [0, n)
    #n is a random integer, chosen from [1, len(S)]
    indexes = list(range(0, random.randint(1, len(S))))
    #now ready to start building mapping; initialize it to empty list
    mapping = []
    #we want to ensure that all elements of indexes are mapped to at least once
    #otherwise certain mappings become artificially exceedingly rare, like the mapping that corresponds to equality
    #so, the first element of S maps to the first element of indexes,
    #the second element of S maps to the second element of indexes, and so on through the last element of indexes
    for i in indexes:
        mapping.append((S[i], i))
    #remember that len(indexes) <= len(S)
    #so for any element of S not yet mapped, just choose an image from indexes randomly
    for s in S[len(indexes):]:
        ix = random.choice(indexes)
        mapping.append((s, ix))
    return mapping 



def find_equivalence_relation_from_mapping(f):
    "Given a mapping, return the corresponding equivalence relation"
    p = find_partition_from_mapping(f)
    return find_equivalence_relation_from_partition(p)



def calculate_stirling(n, k):
    x = 1/math.factorial(k)
    s = 0
    for j in range(k+1):
        a = (-1)**(k-j)
        b = j**n
        c = math.factorial(k)/(math.factorial(j)*math.factorial(k-j))
        s = s + a*b*c
    return round(x*s)


def calculate_bell(n):
    s = 0
    for k in range(1,n+1):
        s = s + calculate_stirling(n, k)
    return s



def make_refinement(Rprime):
    "Given and equivalence relation Rprime, make a random refinement"
    Pprime = find_partition_from_equivalence_relation(Rprime)
    #first decide how many and then which equivalence classes to split
    number_to_split = random.randint(1, len(Pprime))
    to_split = random.sample(Pprime, number_to_split)
    #for each equivalence class, create new equivalence relation with members of that class
    #this is basis for new Rprime, which refines R
    P = copy.deepcopy(Pprime)
    for ec in to_split:
        if len(ec) == 1:
            continue
        P.remove(ec)
        r = make_random_equivalence_relation(ec)
        p = find_partition_from_equivalence_relation(r)
        for b in p:
            P.append(b)
    P.sort()
    R = find_equivalence_relation_from_partition(P)
    return R



def make_sigma(S_R, S_Rprime):
    "Given quotient set of refining relation and quotient set of relation that is refined, make mapping sigma between them"
    sigma_redundant = []
    for b1 in S_R:
        for b2 in S_Rprime:
            if b1[0] in b2:
                sigma_redundant.append((b1, b2))
                break
    sigma = remove_redundant_items(sigma_redundant)
    sigma.sort()
    return sigma


def calculate_maximal_equivalence_relation(S):
    "Given a set S, calculate the maximal equivalence relation on the set (all of S x S)"
    R = []
    for s1 in S:
        for s2 in S:
            R.append((s1, s2))
    R.sort()
    return R



def calculate_minimal_equivalence_relation(S):
    "Given a set S, calculate the minimal equivalence relation on the set (equality)"
    R = []
    for s in S:
        R.append((s, s))
    R.sort()
    return R 



def make_set_of_equivalence_relations(S, n):
    "Given a set S and an positive integer n, make n equivalence relations on the set S and return as a dict"
    collection = {}
    for i in range(n):
        R = make_random_equivalence_relation(S)
        collection[i] = R
    return collection 



def calculate_intersection_of_set_of_equivalence_relations(collection):
    "Given a collection of equivalence relations as a dict, calculate the intersection of all equivalence relations in the collection"
    S = get_underlying_elements_from_relation(collection[0])
    inter = calculate_maximal_equivalence_relation(S)
    for i in collection:
        inter = calculate_er_intersection(inter, collection[i])    
    return inter 



def calculate_intersection_and_remainders(s1, s2):
    "Given 2 set like objects, calculate their intersection and remainder in s1 and s2 that is not in intersection"
    left_remainder = []
    intersection = []
    right_remainder = []
    for e1 in s1:
        if e1 in s2:
            intersection.append(e1)
        else:
            left_remainder.append(e1)
    for e2 in s2:
        if e2 not in s1:
            right_remainder.append(e2)
    intersection.sort()
    left_remainder.sort()
    right_remainder.sort()
    return left_remainder, intersection, right_remainder 



def make_random_corresponding_factor(R_target, R_factor1):
    "Given a target ER and a factor ER (factor1), find another ER factor2 such that factor1 int factor2 = target"

    P_target = find_partition_from_equivalence_relation(R_target)
    P_factor1 = find_partition_from_equivalence_relation(R_factor1)

    P_factor1_copy = P_factor1.copy()  
    separate = []

    #we look at factor1 and target to see which blocks of factor1 must be split by the complementary relation
    for b1 in P_factor1: 
        for b2 in P_target: 
            lr, i, rr = calculate_intersection_and_remainders(b1, b2)
            #if intersection is empty, we have no evidence to split this block (keep checking for all b2 in P_target)
            if i == []:
                continue
            else:
                #by construction, if intersection is not empty, the right remainder should always be empty, 
                #since P_factor1 is refined by P_target, so this is really a check; TODO: make this an assert
                if rr != []:
                    print('{} {} -> {} {} {}'.format(b1, b2, lr, i, rr))
                if lr != []:
                    #put the intersection into separate (i.e., things that get separated)
                    separate.append(i)
                    #don't need to put the left remainder into separate, 
                    #it will actually be the intersection of another comparison
                    try:
                        P_factor1_copy.remove(b1)
                    except:
                        pass
                    
    P_factor2 = []
    #now separate the blocks that need to be separated
    #can either join the block fragments to other blocks, or put them by themselves
    for b in separate:
        join_ix = random.randint(0, len(P_factor1_copy))
        try:
            P_factor1_copy[join_ix] = P_factor1_copy[join_ix] + b
            P_factor2.append(P_factor1_copy[join_ix])
            P_factor1_copy.remove(P_factor1_copy[join_ix])
        except:
            P_factor2.append(b)
    for b in P_factor1_copy:
        P_factor2.append(b)
    P_factor2.sort()

    R_factor2 = find_equivalence_relation_from_partition(P_factor2)

    return R_factor2




def make_unrefinement(Rprime):
    "Given and equivalence relation Rprime, make a relation such that Rprime is a refinement of R"
    Pprime = find_partition_from_equivalence_relation(Rprime)
    number_to_join = random.randint(1, len(Pprime))
    to_join = random.sample(Pprime, number_to_join)
    P = []
    new_block = []
    for b in to_join:
        new_block = new_block + b
        Pprime.remove(b)
    new_block.sort()
    Pprime.append(new_block)
    Pprime.sort()
    R = find_equivalence_relation_from_partition(Pprime)
    return R 
    


def make_collection_of_factors(R, random_size=True, remove_redundant_factors=True):
    "Given an equivalence relation R, make a collection of equivalence relations which, when intersected, give R"
    R_factor1 = make_unrefinement(R)
    R_factor2 = make_random_corresponding_factor(R, R_factor1)
    collection = [R_factor1, R_factor2]
    if random_size:
        number_factors = random.randint(2,5)
    else:
        number_factors = 2
    for i in range(number_factors):
        R_factor = random.choice(collection)
        collection.remove(R_factor)
        R_factor1 = make_unrefinement(R_factor)
        R_factor2 = make_random_corresponding_factor(R_factor, R_factor1)
        collection.append(R_factor1)
        collection.append(R_factor2)
    if remove_redundant_factors:
        collection = remove_redundant_items(collection)
    return collection    



def check_compatible(T, R):
    "Given an automorphism T and an equivalence relation R, check whether T is compatible with R"
    compatible = True
    S = get_underlying_elements_from_relation(R)
    for s1 in S:
        for s2 in S:
            if s1_related_to_s2(s1, s2, R):
                ts1 = evaluate_mapping(T, s1)
                ts2 = evaluate_mapping(T, s2)
                if not s1_related_to_s2(ts1, ts2, R):
                    compatible = False
                    break
    return compatible




def make_t_transform(T, R):
    "Given an automorphism T, and an equivalence relation R, find Rt, the T-transform of R"
    Rt = []
    S = get_underlying_elements_from_relation(R)
    for s1 in S:
        for s2 in S:
            ts1 = evaluate_mapping(T, s1)
            ts2 = evaluate_mapping(T, s2)
            if s1_related_to_s2(ts1, ts2, R):
                Rt.append((s1, s2))
    Rt = remove_redundant_items(Rt)
    Rt.sort()
    return Rt    



def make_tbars(R):
    "Given an equivalence relation R, make all the automorphisms (Tbars) on S/R"
    P = find_partition_from_equivalence_relation(R)
    Tbars_initial = make_automorphisms(P)
    Tbars = []
    for tbar in Tbars_initial:
        ok = True
        for x, y in tbar:
            if len(x) != len(y):
                ok = False
        if ok:
            Tbars.append(tbar)
    return Tbars

def make_random_compatible_automorphism(R):
    "Given an equivalence relation R, randomly make an automorphism T with which it is compatible"
    Tbars = make_tbars(R)
    tbar = random.choice(Tbars)
    T = []
    for x, y in tbar:
        #TO DO: DON'T NEED TO MAKE ALL AUTOMORPHISMS OF Y, CAN JUST SHUFFLE TO MAKE ONE RANDOMLY
        ybag = make_automorphisms(y)
        yq = random.choice(ybag)
        for i, e in enumerate(x):
            T.append((x[i], yq[i][1]))
    T.sort()
    return T



def make_all_compatible_automorphisms(R):
    "Given an equivalence relation R, make set of all automorphisms that are compatible with R, Pr"

    #step 1: get partition from R, then make a list of all elements in order; this is domain for all automorphisms
    P = find_partition_from_equivalence_relation(R)
    X = []
    for p in P:
        X = X + p
    
    #step 2: make all possible automorphisms on S/R, i.e., all possible Tbar mappings
    Tbars = make_tbars(R)

    #step 3: now cycle through all possible Tbars
    compat_autos = []
    for tbar in Tbars[:]:
        #this temporary dictionary has 0, 1, 2, ... as keys, and a particular permutation of a partition as values
        d = {}
        for i, (x, y) in enumerate(tbar):
            #x is a 'source equivalence class', y is the 'target equivalence class'
            #now make all the permutations of y, containing the elements of the target equivalence class 
            perms = []
            for yq in itertools.permutations(y):
                perms.append(yq)
            #now add the integer label for this equivalence class and the permutations in the temporary dictionary
            d[i] = perms
        #now make an array where the n-th item is the list of permutations for the n-th equivalence class
        q = [d[i] for i in d]
        #now using itertools, make the product of all the items in q (use *q to unpack the items)
        for permut in itertools.product(*q):
            #permut is a 2-D matrix, flatten this to 1-D to make the range of the automorphism
            Y = []
            for p in permut:
                Y = Y + list(p)
            #now everything is 'lined up', can match each element in the original X order to an element in Y,
            #this forms an automorphism; once done add this to the list of all compatible automorphisms
            T = []
            for ix, item in enumerate(X):
                T.append((X[ix], Y[ix]))
            T.sort()
            compat_autos.append(T)
    
    return compat_autos    



def make_all_compatible_equivalence_relations(T):
    "Given an automorphism T on an underlying set, find all the equivalence relations with which T is compatible"
    S = get_underlying_elements_from_relation(T)
    equivalence_relations = make_equivalence_relations2(S)
    Pt = []
    for R in equivalence_relations:
        if check_compatible(T, R):
            Pt.append(R)
    Pt.sort()
    return Pt




def make_linearly_ordered_subset(Pt, n):
    "Given the set of equivalence relations with which T is compatible, Pt, and a number, n, make a linearly ordered subset Z in Pt"
    #start by drawing 2 random equivalence relations from Pt, 
    #keep doing this until you have 2 that can be totally ordered with respect to each other
    n = 3
    r1, r2 = random.sample(Pt, 2)
    while not er1_refines_er2(r1, r2) and not er1_refines_er2(r2, r1):
        r1, r2 = random.sample(Pt, 2)
    #add these 2 relations to Z; they are like the 'seed' for Z
    Z = [r1, r2]
    #now randomly draw an additional equivalence relation from Pt
    #if this new relation can be added to Z such that you have a totally ordered set, add to Z
    #keep going until Z reaches specified length (number of elements)
    while len(Z) < n:
        ri = random.choice(Pt)
        #will accept ri if it forms a total order with all the other r in Z; otherwise reject
        accept = True
        for r in Z:
            if not er1_refines_er2(r, ri) and not er1_refines_er2(ri, r):
                accept = False
        if accept:
            Z.append(ri)
        #since we sample with replacement, remove any redundant R
        Z = remove_redundant_items(Z)
    Z.sort()
    return Z



def find_max_and_min_elements(Pt):
    "Given the set of equivalence relations with which T is compatible, Pt, find the set of maximal and minimal elements"
    S = get_underlying_elements_from_relation(Pt[0])
    equality_R = calculate_minimal_equivalence_relation(S)
    trivial_R = calculate_maximal_equivalence_relation(S)
    max_elements, min_elements = [], []
    for r1 in Pt:
        maximal_element = True
        minimal_element = True
        for r2 in Pt:
            if r1 == r2:
                continue
            if er1_refines_er2(r2, r1) and r2 != equality_R:
                minimal_element = False
            if er1_refines_er2(r1, r2) and r2 != trivial_R:
                maximal_element = False
        if maximal_element and r1 != trivial_R:
            max_elements.append(r1)
        if minimal_element and r1 != equality_R:
            min_elements.append(r1)   
    return max_elements, min_elements 



def make_T_n(T, n):
    "Given an automorphism T and an integer n, make T^n, i.e., the automorphism to the nth power"
    automorphism = T
    if n == 0:
        return T
    power, step = 0, 1
    if n < 0:
        step = -1
    while abs(power) < abs(n):
        automorphism = compose_mappings(T, automorphism)
        power = power + step
    return automorphism    
