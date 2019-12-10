from superdifferentiator.forward.bgfs import bgfs



def test_bgfs():
    
    f = lambda x: (x[0]-1.555)**2 + (x[1]-4)**2
    init_x = [1,2]
    results = bgfs(f,init_x)
    
    
    assert results[0] - 1.555 < 1e-8
    assert results[1] - 4< 1e-8

    
