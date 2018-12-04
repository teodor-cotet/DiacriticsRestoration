import { Component, OnInit } from '@angular/core';
import { ClusteringModel, Cluster } from './clustering.model';
import { ClusterElementType } from './cluster-element-type.data';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

declare var require: any

const ELEMENT_TYPE_CANDIDATE = 0;
const ELEMENT_TYPE_USER_ANSWER = 1;

@Component({
  selector: 'app-clustering',
  templateUrl: './clustering.component.html',
  styleUrls: ['./clustering.component.css']
})
export class ClusteringComponent implements OnInit {

  public serverUrl: string = 'http://141.85.232.83:8080/clustering';
  public title: string;
  public description: string;
  loading: boolean;
  clustering: ClusteringModel;
  clusterErrors: string[];
  minAnswers: number;
  public inputData: string;

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.title = 'Clustering';
    this.description = 'This page shows generated clusters using Natural Language Processing techniques. Each cluster may contain elements that might be either candidate options or user answers.'
    this.loading = true;
    this.minAnswers = 5;
    this.inputData = this.loadDummyInputData();
    this.clustering = this.loadDummyData();
    this.displayClusteringData();
  }

  loadDummyInputData() {
    return JSON.stringify(require('../../assets/sample-input.json'));
  }

  loadDummyData() {
    let data = require('../../assets/sample-output.json');
    console.log(data);
    return data;
  }

  parseData() {
    this.sendData(this.inputData).subscribe(hero => function (response) {
      console.log('Received response ', response);
      this.clustering = response;
    });
  }

  displayClusteringData() {
    for (var cluster in this.clustering) {
      console.log(cluster);
    }
    this.clusterErrors = new Array();
    var _this = this;
    console.log(this.clustering);
    this.clustering.clusters.forEach(function (cluster, key) {
      var elements: any = cluster;
      var localCluster = new Cluster(elements);
      var errorCode = _this.errorCode(localCluster, _this.minAnswers);
      console.log('Error code ' + errorCode);
      _this.clusterErrors[key] = _this.errorToString(errorCode);
    });
    if (this.clustering.clusters.length > 0) this.loading = false;
    console.log(this.clustering);
  }

  sendData(data): Observable<any> {
    const httpOptions = {
      headers: new HttpHeaders({
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      })
    };
    return this.http.post(this.serverUrl, data, httpOptions)
      .pipe(
        catchError(this.handleError)
      );
  }

  handleError(error: HttpErrorResponse) {
    if (error.error instanceof ErrorEvent) {
      // A client-side or network error occurred. Handle it accordingly.
      console.error('An error occurred:', error.error.message);
    } else {
      // The backend returned an unsuccessful response code.
      // The response body may contain clues as to what went wrong,
      console.error(
        `Backend returned code ${error.status}, ` +
        `body was: ${error.error}`);
    }
    // return an observable with a user-facing error message
    return throwError(
      'Something bad happened; please try again later.');
  }

  typeToString(type) {
    return ClusterElementType[type];
  }

  errorCode(cluster: Cluster, minAnswers: number = 0) {
    var noCandidates = 0;
    var noAnswers = 0;
    console.log(cluster);
    var errorType = 0;
    cluster.elements.forEach(function (element) {
      console.log('Element ', element);
      if (element.type == ELEMENT_TYPE_CANDIDATE) noCandidates++;
      else noAnswers++;
      console.log(noAnswers);
      if (noCandidates >= 2) {
        errorType = 1;
        return;
      }
    });
    if (errorType != 0) return errorType;
    if (noCandidates == 0 && noAnswers <= minAnswers) return 2;
    return 0;
  }

  errorToString(error: number) {
    switch (error) {
      case 1: return 'You should separate candidate options for this cluster!';
      case 2: return 'Its recommended to introduce a candidate option that covers this cluster!';
      default: return '';
    }
  }

  getClusterError(key) {
    return this.clusterErrors[key];
  }

}
