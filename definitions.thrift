namespace java thrift
namespace py definitions

//Service for Semantic Queries.
//----------------------------------------------
// Notice that both functions in this service are blocking functions. There is no callback defined in the MmtService for sending the result. 
// This is because the Mmt needs the lists in order to populate the interface with valid data.
//----------------------------------------------
service Word2VecService
{
    oneway void trainModel(1:string path, 2:i32 noEpochs, 3:i32 layerSize),
    map<string, list<double>> loadModel(1:string path),
}