#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Created on Mon Aug 4 2016
Modified on Mon Jan 13 2017
Filename    : LinkingUtil.mingbin.feature.test.2016.py
Description : prepare test data for KBP EL
Author      : fwei
"""


import numpy, os, codecs, itertools, logging, math
import scipy.sparse
from gigaword2feature import *
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from collections import defaultdict

logger = logging.getLogger( __name__ )

# Global Variable
offsets_of_original_list = []
document_id = []
entity_type = []
mention_type = []

def LoadED( rspecifier, language = 'eng' ):

    entity2cls = {  # KBP2015 label
                    'PER_NAM' : 0, 
                    'ORG_NAM' : 1, 
                    'GPE_NAM' : 2, 
                    'LOC_NAM' : 3, 
                    'FAC_NAM' : 4, 
                    'PER_NOM' : 5, 
                    'TTL_NAM' : 5,
                    
                    # KBP2016
                    'PER_NAME' : 0,  
                    'ORG_NAME' : 1, 
                    'GPE_NAME' : 2, 
                    'LOC_NAME' : 3, 
                    'FAC_NAME' : 4, 
                    'PER_NOMINAL' : 5,
                    'ORG_NOM' : 6,
                    'GPE_NOM' : 7,
                    'LOC_NOM' : 8,
                    'FAC_NOM' : 9,
                    'TITLE_NAME' : 5,
                    'TITLE_NOMINAL' : 5
                    
                } 

    if os.path.isfile( rspecifier ):
        with codecs.open( rspecifier, 'rb', 'utf8' ) as fp:
            processed, original = fp.read().split( u'=' * 128, 1 )
            original = original.strip()

            # texts, tags, failures = processed.split( u'\n\n\n', 2 )
            texts = processed.split( u'\n\n\n' )[0]
            
            
            for text in texts.split( u'\n\n' ):
                parts = text.split( u'\n' )
                # assert len(parts) in [2, 3], 'sentence, offsets, labels(optional)'
                if len( parts ) not in [2, 3]:
                    logger.exception( text )
                    continue

                sent, boe, eoe, target, mids, spelling = parts[0].split(u' '), [], [], [], [], []
                offsets = map( lambda x : (int(x[0]), int(x[1])),
                               [ offsets[1:-1].split(u',') for offsets in parts[1].split() ] )
                assert len(offsets) == len(sent), rspecifier + '\n' + \
                        str( offsets ) + '\n' + str( sent ) + '\n%d vs %d' % (len(offsets), len(sent))
                
                                
                if len(parts) == 3:
                    for ans in parts[-1].split():
                        try:
                            begin_idx, end_idx, mid, mention1, mention2 = ans[1:-1].split(u',')
                            target.append( entity2cls[str(mention1 + u'_' + mention2)] )
                            boe.append( int(begin_idx) )
                            eoe.append( int(end_idx) )
                            mids.append( mid )
                            offsets_of_original_list.append("{0}-{1}".format(offsets[boe[-1]][0],
                                            offsets[eoe[-1] - 1][1] - 1))
                            #print offsets_of_original_list
                            spelling.append( original[ offsets[boe[-1]][0] : offsets[eoe[-1] - 1][1] ] )
                            #print spelling
                            document_id.append( rspecifier.split('/')[-1] )
                            #print document_id
                            #exit(0)
                        except ValueError as ex1:
                            logger.exception( rspecifier )
                            logger.exception( ans )
                        except KeyError as ex2:
                            logger.exception( rspecifier )
                            logger.exception( ans )

                        try:
                            assert 0 <= boe[-1] < eoe[-1] <= len(sent), \
                                    '%s  %d  ' % (rspecifier.split('/')[-1], len(sent)) + \
                                    '  '.join( str(x) for x in [sent, boe, eoe, target, mids] )
                        except IndexError as ex:
                            logger.exception( rspecifier )
                            logger.exception( str(boe) + '   ' + str(eoe) )
                            continue
                    assert( len(boe) == len(eoe) == len(target) == len(mids) )

                # move this part to processed_sentence
                # if language == 'eng':
                #     for i,w in enumerate( sent ):
                #         sent[i] = u''.join( c if 0 <= ord(c) < 128 else chr(0) for c in list(w) )
                yield sent, boe, eoe, target, mids, spelling


    else:
        for filename in os.listdir( rspecifier ):
            for X in LoadED( os.path.join( rspecifier, filename ), language ):
                yield X


def LoadEL( rspecifier, language = 'eng', window = 1 ):
    if os.path.isfile( rspecifier ):
        data = list( LoadED( rspecifier, language ) )
        for i,(sent,boe,eoe,label,mid,spelling) in enumerate(data):
            if len(label) > 0:
                previous, next = [], []
                for s,_,_,_,_,_ in data[i - window: i]:
                    previous.extend( s )
                for s,_,_,_,_,_ in data[i + 1: i + 1 + window]:
                    next.extend( s )
                yield previous + sent + next, \
                      [ len(previous) + b for b in boe ], \
                      [ len(previous) + e for e in eoe ], \
                      label, mid, spelling

    else:
        for filename in os.listdir( rspecifier ):
            for X in LoadEL( os.path.join( rspecifier, filename ), language ):
                yield X



def PositiveEL( embedding_basename,
                rspecifier, language = 'eng', window = 1 ):

    raw_data = list( LoadEL( rspecifier, language, window ) )

    # with open( embedding_basename + '.word2vec', 'rb' ) as fp:
    #   shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
    #   projection = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
    # logger.debug( 'embedding loaded' )

    with codecs.open( embedding_basename + '.wordlist', 'rb', 'utf8' ) as fp:
        n_word = len( fp.read().strip().split() )
    logger.debug( 'a vocabulary of %d words is used' % n_word )

    numericizer = vocabulary( embedding_basename + '.wordlist',
                              case_sensitive = False )

    bc = batch_constructor( [ rd[:4] for rd in raw_data ],
                              numericizer, numericizer,
                              window = 1024, n_label_type = 7 )
    logger.debug( bc )

    index_filter = set([2, 3, 6, 7, 8])

    mid_itr = itertools.chain.from_iterable( rd[-2] for rd in raw_data )

    mention = itertools.chain.from_iterable( rd[-1] for rd in raw_data )

    # for sent, boe, eoe, _, _ in raw_data:
    #   for b,e in zip( boe, eoe ):
    #       mention.append( sent[b:e] )

    # feature_itr = bc.mini_batch( 1,
    #                            shuffle_needed = False,
    #                            overlap_rate = 0, disjoint_rate = 0,
    #                            feature_choice = 7  )
    # # assert( len(list(mid_itr)) == len(list(feature_itr)) )

    # for mid, feature in zip( mid_itr, feature_itr ):
    #   yield mid, \
    #         [ f.reshape([-1])[1::2] if i in index_filter else f.reshape([-1]) \
    #           for i,f in enumerate(feature[:9]) ]

    l1v, r1v, l1i, r1i, l2v, r2v, l2i, r2i, bow = \
            bc.mini_batch( len(bc.positive),
                           shuffle_needed = False,
                           overlap_rate = 0,
                           disjoint_rate = 0,
                           feature_choice = 7  ).next()[:9]
    l1 = csr_matrix( ( l1v, ( l1i[:,0].reshape([-1]), l1i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    l2 = csr_matrix( ( l2v, ( l2i[:,0].reshape([-1]), l2i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    r1 = csr_matrix( ( r1v, ( r1i[:,0].reshape([-1]), r1i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    r2 = csr_matrix( ( r2v, ( r2i[:,0].reshape([-1]), r2i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    bow = csr_matrix( ( numpy.ones( bow.shape[0] ),
                        ( bow[:,0].reshape([-1]), bow[:,1].reshape([-1]) ) ),
                      shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    return list(mid_itr), mention, l1, l2, r1, r2, bow




def LoadTfidf( tfidf_basename,  col ):

    with open( tfidf_basename + '.list' ) as fp:
        idx2mid = [ mid[1:-1] for mid in fp.read().strip().split() ]
        mid2idx = { m:i for i,m in enumerate( idx2mid ) }

    indices = numpy.fromfile( tfidf_basename + '.indices', dtype = numpy.int32 )
    data = numpy.fromfile( tfidf_basename + '.data', dtype = numpy.float32 )
    indptr = numpy.fromfile( tfidf_basename + '.indptr', dtype = numpy.int32 )
    assert indices.shape == data.shape

    mid2tfidf = csr_matrix( (data, indices, indptr),
                            shape = (indptr.shape[0] - 1, col) )
    del data, indices, indptr
    mid2tfidf = mid2tfidf.astype( numpy.float32 )
    mid2tfidf.sort_indices()
#   with open( tfidf_basename + '.list' ) as fp:
#       idx2mid = [ mid[1:-1] for mid in fp.read().strip().split() ]
#       mid2idx = { m:i for i,m in enumerate( idx2mid ) }

    return mid2tfidf, idx2mid, mid2idx


def LoadCandiDict( candi_filename ):
    candi_item_dict = {}
      # open candidate file
    with open( candi_filename, 'rb') as candifile:
        for cline in candifile.readlines():
            array = cline.split('\t')
#            if(array[4] != ''):         # remove the NIL items
            if(True):
                candi_item = array[5]
                candi_item = re.sub('[|\[\]]', '', candi_item)
                candi_item = candi_item.split('  ')

                candi_item = [word.replace('.', '/') for word in candi_item]   # replace '.' with '/' in mid
#               candi_item_list.append(candi_item)

                candi_item = map(lambda s: s.strip(), candi_item)
                candi_item_dict[array[0]] = candi_item

    logger.info( 'candidate loaded' )
    return candi_item_dict

if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s',
                         level = logging.DEBUG )

    embedding_basename = '/eecs/research/asr/mingbin/cleaner/word2vec/gigaword/gigaword128-case-insensitive'
    tfidf_basename = '/eecs/research/asr/Shared/Entity_Linking_training_data_from_Freebase/result/FBeasy/mid2tfidf'
    candi_filename = '/local/scratch/fwei/KBP/EL/data/out.edl2run3.candidate.txt'
    output_dir = '/local/scratch/fwei/KBP/EL/result/2016'
    input_dir = '/local/scratch/fwei/KBP/EL/xml_output/2016'


    window_fofe = 1 # how many sentences before or behind the target sentence


    with open( embedding_basename + '.word2vec', 'rb' ) as fp:
        shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
        projection = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
    logger.info( 'embedding loaded' )

    mid2tfidf, idx2mid, mid2idx = LoadTfidf( tfidf_basename, projection.shape[0] )
    logger.info( 'tfidf loaded' )


    solution, mention, l1, l2, r1, r2, bow = PositiveEL( embedding_basename, input_dir,  window = window_fofe )
    logger.info( 'fofe loaded' )

    candi_item_dict = LoadCandiDict( candi_filename )

    f_test_kb = open(os.path.join( output_dir, 'no_sorted_uniq_test.kb'), 'w')
    f_test_pair = open(os.path.join( output_dir, 'no_shuf_test.pair'), 'w')

    print 'r2: ' + str(r2.shape[0])
    print 'offset list: ' + str(len(offsets_of_original_list))
    exit(0)

    # test.ment & test.fb & test.pair & test.map
    with open(os.path.join( output_dir, 'test.ment'), 'w') as f_test_ment,  \
    codecs.open(os.path.join( output_dir, 'test.mentMap'), 'w', encoding='utf-8') as f_test_map:
        for t, (s, m) in enumerate( zip (solution, mention) ):
            # variable
            numCopyPos = 0
            
            # test.map
            f_test_map.write('\t'.join([str(t), u':'.join([document_id[t], offsets_of_original_list[t]]), m.replace('\n',' ')]) + '\n')         
            
            # test.ment
            strL2 =  ' '.join('%s,%s' % x for x in zip(l2[t].indices, l2[t].data))
            strR2 =  ' '.join('%s,%s' % x for x in zip(r2[t].indices, r2[t].data))
            strBow =  ' '.join('%s,%s' % x for x in zip(bow[t].indices, bow[t].data / bow[t].data.shape[0]))

            f_test_ment.write('\t'.join([str(t), strL2, strR2, strBow]) + '\n')


            # negative  -->  test.fb & test.pair
            query_feature =  u':'.join([document_id[t], offsets_of_original_list[t]])
            if query_feature in candi_item_dict:
                candi_list = candi_item_dict.get(query_feature)

#            candi_idx_list = [ mid2idx[c] for c in candi_list if c in mid2idx and c !=  s.replace('.','/')] # judge if true label
                                                                                                            # in the candidate list
            candi_idx_list = [ mid2idx[c] for c in candi_list if c in mid2idx ]
            numCopyPos = math.floor(len(candi_idx_list) / 2)
            for candi in candi_idx_list:
#                assert idx2mid[candi].replace('/','.') <> s  # assure true lable is not in the candidate list
                negFea = mid2tfidf[candi]
                strNeg = ' '.join('%s,%s' % x for x in zip(negFea.indices, negFea.data))
                f_test_kb.write('\t'.join([idx2mid[candi].replace('/','.'), strNeg]) + '\n')
                f_test_pair.write('\t'.join([str(t), idx2mid[candi].replace('/','.'), '0']) + '\n')


            # positive  -->  test.fb & test.pair
#            if s.replace('.','/') in mid2idx:
#                posFea = mid2tfidf[mid2idx[s.replace('.','/')]]
#                strPos = ' '.join('%s,%s' % x for x in zip(posFea.indices, posFea.data))
#                f_train_fb.write('\t'.join([s, strPos]) + '\n')
#                for i in range(int(numCopyPos) + 1):
#                    f_train_pair.write('\t'.join([str(t), s, '1']) + '\n')

    f_test_kb.close()
    f_test_pair.close()

    # remove duplicate from .kb
    os.system('cat ' + os.path.join( output_dir, 'no_sorted_uniq_test.kb') + ' | sort | uniq > ' + \
                os.path.join( output_dir, 'test.kb') )

    # shuffle .pair
    os.system('cat ' + os.path.join( output_dir, 'no_shuf_test.pair') + ' | perl -MList::Util=shuffle -e \'print shuffle(<STDIN>);\' > ' + \
                os.path.join( output_dir, 'test.pair') )

    # move cache file to tmp
    os.system('mv ' + os.path.join( output_dir, 'no_shuf_test.pair ') + os.path.join( output_dir, 'no_sorted_uniq_test.kb') + ' /tmp')
